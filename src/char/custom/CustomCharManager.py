import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, RLock, Thread
from typing import TYPE_CHECKING

import cv2
import numpy as np
from ok import Logger, og

import src.char.custom.CustomCharDb as CustomCharDb
from src.Labels import Labels

if TYPE_CHECKING:
    from src.combat.BaseCombatTask import BaseCombatTask

logger = Logger.get_logger(__name__)

CUSTOM_CHARS_DIR = "custom_chars"
FEATURES_DIR = os.path.join(CUSTOM_CHARS_DIR, "features")
DB_PATH = os.path.join(CUSTOM_CHARS_DIR, "db.json")
DB_SCHEMA_VERSION = CustomCharDb.DB_SCHEMA_VERSION


class CustomCharManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(CustomCharManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return
        self._data_lock = RLock()
        os.makedirs(FEATURES_DIR, exist_ok=True)
        self.db = CustomCharDb.default_db()
        self._feature_cache = {}
        self._raw_feature_cache = {}
        self._cache_mask = None
        self._cache_scr_w = -1
        self._cache_scr_h = -1
        self._cache_fids = set()
        self._preheat_started = False
        self.load_db()
        self.migrate_db_schema()
        self.validate_db()
        self.initialized = True
        self.preheat_feature_cache_async()

    @staticmethod
    def _as_text(value) -> str:
        return CustomCharDb.as_text(value)

    @classmethod
    def _is_blank_text(cls, value) -> bool:
        return CustomCharDb.is_blank_text(value)

    @staticmethod
    def _builtin_entries() -> dict:
        from src.char.CharFactory import char_dict

        return {key: value for key, value in char_dict.items() if key != "char_default"}

    @staticmethod
    def _locale_name() -> str:
        app = getattr(og, "app", None)
        if app and hasattr(app, "locale"):
            try:
                return app.locale.name()
            except Exception:
                return ""
        return ""

    @staticmethod
    def get_builtin_prefix() -> str:
        app = getattr(og, "app", None)
        if app and hasattr(app, "tr"):
            return f"{app.tr('[内置代码]')} "
        return "[内置代码] "

    @classmethod
    def is_builtin_combo(cls, combo_id: str) -> bool:
        return cls._as_text(combo_id) in cls._builtin_entries()

    @classmethod
    def get_builtin_combo_name(cls, combo_id: str) -> str:
        entries = cls._builtin_entries()
        meta = entries.get(cls._as_text(combo_id))
        if not isinstance(meta, dict):
            return cls._as_text(combo_id)
        if cls._locale_name() == "zh_CN" and meta.get("cn_name"):
            return cls._as_text(meta["cn_name"])
        char_cls = meta.get("cls")
        return getattr(char_cls, "__name__", cls._as_text(combo_id))

    @classmethod
    def iter_builtin_combo_items(cls):
        for combo_id in cls._builtin_entries().keys():
            yield cls.get_builtin_combo_name(combo_id), combo_id

    @staticmethod
    def _default_fixed_team():
        return CustomCharDb.default_fixed_team()

    @classmethod
    def _normalize_fixed_team_slot(cls, slot) -> dict:
        return CustomCharDb.normalize_fixed_team_slot(slot)

    @classmethod
    def _normalize_fixed_team_config(cls, config) -> dict:
        return CustomCharDb.normalize_fixed_team_config(config)

    @staticmethod
    def _default_db():
        return CustomCharDb.default_db()

    def _character_name_from_record(self, char_id: str, char_data: dict) -> str:
        return CustomCharDb.character_name_from_record(char_id, char_data)

    def _find_character_id_by_name(self, char_name: str) -> str | None:
        target = self._as_text(char_name).strip()
        if self._is_blank_text(target):
            return None
        for char_id, char_data in self.db.get("characters", {}).items():
            if not isinstance(char_data, dict):
                continue
            if self._character_name_from_record(char_id, char_data) == target:
                return char_id
        return None

    def _generate_character_id(self) -> str:
        while True:
            char_id = f"char_{uuid.uuid4().hex}"
            if char_id not in self.db["characters"]:
                return char_id

    def _generate_combo_id(self, existing_ids: set[str] | None = None) -> str:
        existing_ids = existing_ids or set(self.db.get("combos", {}).keys())
        while True:
            combo_id = f"combo_{uuid.uuid4().hex}"
            if combo_id not in existing_ids and not self.is_builtin_combo(combo_id):
                return combo_id

    def load_db(self):
        with self._data_lock:
            self.db = CustomCharDb.load_db(DB_PATH, logger)

    def validate_db(self):
        with self._data_lock:
            modified = CustomCharDb.validate_db(self.db, FEATURES_DIR, self.is_builtin_combo)

            if modified:
                self._invalidate_feature_cache()
                self.save_db()

    def save_db(self):
        with self._data_lock:
            CustomCharDb.save_db(DB_PATH, self.db, logger)

    def _invalidate_feature_cache(self):
        self._feature_cache.clear()
        self._cache_scr_w = -1
        self._cache_scr_h = -1
        self._cache_fids = set()

    def _invalidate_raw_feature_cache(self, feature_id=None):
        if feature_id is None:
            self._raw_feature_cache.clear()
        else:
            self._raw_feature_cache.pop(feature_id, None)

    def _get_feature_ids_snapshot(self):
        with self._data_lock:
            feature_ids = set(self.db.get("features", {}).keys())
            for char_data in self.db.get("characters", {}).values():
                if isinstance(char_data, dict):
                    feature_ids.update(char_data.get("feature_ids", []))
            return list(feature_ids)

    def preheat_feature_cache(self):
        feature_ids = self._get_feature_ids_snapshot()
        if not feature_ids:
            return

        worker_count = min(8, len(feature_ids))
        if worker_count == 1:
            for fid in feature_ids:
                self._load_feature_image_cached(fid)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                list(executor.map(self._load_feature_image_cached, feature_ids))
        logger.debug(f"preheated {len(feature_ids)} custom feature images")

    def _preheat_feature_cache_worker(self):
        try:
            self.preheat_feature_cache()
        except Exception as e:
            logger.error("Failed to preheat custom feature images", e)

    def preheat_feature_cache_async(self):
        with self._data_lock:
            if self._preheat_started:
                return
            self._preheat_started = True
        Thread(
            target=self._preheat_feature_cache_worker,
            name="custom-feature-cache-preheat",
            daemon=True,
        ).start()

    def migrate_db_schema(self):
        with self._data_lock:
            self.db, modified = CustomCharDb.migrate_db_schema(
                self.db,
                self.is_builtin_combo,
                self.get_builtin_prefix,
                self.iter_builtin_combo_items,
                self._generate_combo_id,
            )
            if modified:
                self.save_db()

    def find_custom_combo_id_by_name(self, combo_name: str) -> str:
        combo_name = self._as_text(combo_name)
        if self._is_blank_text(combo_name):
            return ""
        for combo_id, combo_data in self.db.get("combos", {}).items():
            if isinstance(combo_data, dict) and combo_data.get("name") == combo_name:
                return combo_id
        return ""

    def add_combo(self, combo_name: str, content: str, combo_id: str | None = None) -> str:
        """Add or update a custom combo and return its stable combo id."""
        with self._data_lock:
            combo_name = self._as_text(combo_name)
            if self._is_blank_text(combo_name):
                return ""

            existing_id = self.find_custom_combo_id_by_name(combo_name)
            combo_id = combo_id or existing_id or self._generate_combo_id()
            if self.is_builtin_combo(combo_id):
                return ""

            self.db["combos"][combo_id] = {
                "name": combo_name,
                "content": self._as_text(content),
            }
            self.save_db()
            return combo_id

    def update_combo(self, combo_id: str, content: str, combo_name: str | None = None) -> bool:
        with self._data_lock:
            combo_id = self._as_text(combo_id)
            if combo_id not in self.db["combos"] or self.is_builtin_combo(combo_id):
                return False
            record = self.db["combos"][combo_id]
            if combo_name is not None and not self._is_blank_text(combo_name):
                record["name"] = self._as_text(combo_name)
            record["content"] = self._as_text(content)
            self.save_db()
            return True

    def delete_combo(self, combo_id: str):
        """删除出招表"""
        with self._data_lock:
            combo_id = self._as_text(combo_id)
            deleted = False
            if combo_id in self.db["combos"]:
                del self.db["combos"][combo_id]
                deleted = True
            fixed_team = self._normalize_fixed_team_config(self.db.get("fixed_team"))
            fixed_team_changed = False
            for slot in fixed_team["slots"]:
                if slot["combo_id"] == combo_id:
                    slot["combo_id"] = ""
                    fixed_team_changed = True
            if fixed_team_changed:
                self.db["fixed_team"] = fixed_team
            if deleted or fixed_team_changed:
                self.save_db()

    def is_custom_combo_exist(self, combo_id: str):
        """判断出招表是否存在"""
        with self._data_lock:
            return self._as_text(combo_id) in self.db["combos"]

    def get_combo(self, combo_id: str):
        """获取出招表"""
        with self._data_lock:
            combo_id = self._as_text(combo_id)
            combo_data = self.db["combos"].get(combo_id)
            if isinstance(combo_data, dict):
                return combo_data.get("content", "")
            return ""

    def get_combo_name(self, combo_id: str, with_builtin_prefix=False) -> str:
        combo_id = self._as_text(combo_id)
        if not combo_id:
            return ""
        if self.is_builtin_combo(combo_id):
            name = self.get_builtin_combo_name(combo_id)
            if with_builtin_prefix:
                return f"{self.get_builtin_prefix()}{name}"
            return name
        combo_data = self.db.get("combos", {}).get(combo_id)
        if isinstance(combo_data, dict):
            return self._as_text(combo_data.get("name", combo_id))
        return combo_id

    def get_all_combos(self):
        with self._data_lock:
            combos = [data["name"] for data in self.db["combos"].values() if isinstance(data, dict)]
            combos.extend([name for name, _ in self.iter_builtin_combo_items()])
            return combos

    def get_all_combo_items(self, with_builtin_prefix=False):
        """
        Return combo options as (name, id) tuples for UI binding.
        """
        with self._data_lock:
            items = []
            for combo_id, data in self.db["combos"].items():
                if isinstance(data, dict):
                    items.append((data.get("name", combo_id), combo_id))
            for combo_name, combo_id in self.iter_builtin_combo_items():
                if with_builtin_prefix:
                    combo_name = f"{self.get_builtin_prefix()}{combo_name}"
                items.append((combo_name, combo_id))
            return items

    def create_character(self, char_name, combo_id) -> str:
        """创建角色并返回 char_id"""
        with self._data_lock:
            char_name = self._as_text(char_name).strip()
            combo_id = self._as_text(combo_id)
            if self._is_blank_text(char_name):
                return ""
            existing_id = self._find_character_id_by_name(char_name)
            if existing_id:
                return existing_id
            if (
                combo_id
                and not self.is_builtin_combo(combo_id)
                and combo_id not in self.db["combos"]
            ):
                combo_id = ""
            char_id = self._generate_character_id()
            self.db["characters"][char_id] = {
                "name": char_name,
                "combo_id": combo_id,
                "feature_ids": [],
            }
            self._invalidate_feature_cache()
            self.save_db()
            return char_id

    def update_character(self, char_id, char_name=None, combo_id=None) -> bool:
        """更新角色名称或出招表"""
        with self._data_lock:
            if char_id not in self.db["characters"]:
                return False
            char_data = self.db["characters"][char_id]
            if char_name is not None:
                char_name = self._as_text(char_name).strip()
                if self._is_blank_text(char_name):
                    return False
                existing_id = self._find_character_id_by_name(char_name)
                if existing_id and existing_id != char_id:
                    return False
                char_data["name"] = char_name
            if combo_id is not None:
                combo_id = self._as_text(combo_id)
                if (
                    combo_id
                    and not self.is_builtin_combo(combo_id)
                    and combo_id not in self.db["combos"]
                ):
                    combo_id = ""
                char_data["combo_id"] = combo_id
            self._invalidate_feature_cache()
            self.save_db()
            return True

    def delete_character(self, char_id: str):
        """删除角色及其所有特征图，不影响出招表"""
        with self._data_lock:
            if char_id not in self.db["characters"]:
                return
            feature_ids = self.db["characters"][char_id].get("feature_ids", [])
            for fid in feature_ids:
                self.delete_feature_image(fid)
            del self.db["characters"][char_id]
            fixed_team = self._normalize_fixed_team_config(self.db.get("fixed_team"))
            fixed_team_changed = False
            for slot in fixed_team["slots"]:
                if slot.get("char_id") == char_id:
                    slot["char_id"] = ""
                    slot["combo_id"] = ""
                    fixed_team_changed = True
            if fixed_team_changed:
                self.db["fixed_team"] = fixed_team
            self._invalidate_feature_cache()
            self.save_db()



    def add_feature_to_character(self, char_id: str, image_mat, width=0, height=0):
        """为角色保存一张截图并关联特征 UUID"""
        with self._data_lock:
            if char_id not in self.db["characters"]:
                return ""
            fid = f"feat_{uuid.uuid4().hex}"
            self.save_feature_image(fid, image_mat)

            if "features" not in self.db:
                self.db["features"] = {}
            self.db["features"][fid] = {"width": width, "height": height}

            if "feature_ids" not in self.db["characters"][char_id]:
                self.db["characters"][char_id]["feature_ids"] = []

            self.db["characters"][char_id]["feature_ids"].append(fid)
            self._invalidate_feature_cache()
            self.save_db()
            return fid

    def remove_feature_from_character(self, char_id: str, feature_id: str):
        """从角色中移除某个特征"""
        with self._data_lock:
            if char_id not in self.db["characters"]:
                return
            feature_ids = self.db["characters"][char_id].get("feature_ids", [])
            if feature_id in feature_ids:
                feature_ids.remove(feature_id)
                self.delete_feature_image(feature_id)
                self._invalidate_feature_cache()
                self.save_db()

    def save_feature_image(self, feature_id, image_mat):
        """保存特征图"""
        path = os.path.join(FEATURES_DIR, f"{feature_id}.png")
        ok = cv2.imwrite(path, image_mat)
        if not ok:
            raise IOError(f"Failed to write feature image: {path}")
        with self._data_lock:
            self._invalidate_raw_feature_cache(feature_id)

    def delete_feature_image(self, feature_id):
        """删除特征图文件并移除 DB 内独立的特征分辨率记录"""
        with self._data_lock:
            if "features" in self.db and feature_id in self.db["features"]:
                del self.db["features"][feature_id]
            path = os.path.join(FEATURES_DIR, f"{feature_id}.png")
            if os.path.exists(path):
                os.remove(path)
            self._invalidate_raw_feature_cache(feature_id)

    def _load_feature_image_cached(self, feature_id):
        """读取特征图以及其原始分辨率"""
        path = os.path.join(FEATURES_DIR, f"{feature_id}.png")
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            with self._data_lock:
                self._invalidate_raw_feature_cache(feature_id)
            return None, 0, 0

        cache_key = (stat.st_mtime_ns, stat.st_size)
        with self._data_lock:
            cached = self._raw_feature_cache.get(feature_id)
            if cached and cached[0] == cache_key:
                return cached[1], cached[2], cached[3]
            feat_info = self.db.get("features", {}).get(feature_id, {})
            w = feat_info.get("width", 0)
            h = feat_info.get("height", 0)

        mat = cv2.imread(path)
        if mat is None:
            return None, 0, 0

        with self._data_lock:
            self._raw_feature_cache[feature_id] = (cache_key, mat, w, h)
        return mat, w, h

    def load_feature_image(self, feature_id):
        """读取特征图以及其原始分辨率"""
        mat, w, h = self._load_feature_image_cached(feature_id)
        return (mat.copy() if mat is not None else None), w, h

    def _load_resized_feature(self, char_id, feature_id, current_scr_w, current_scr_h):
        saved_img, w, h = self._load_feature_image_cached(feature_id)
        if saved_img is None:
            return char_id, feature_id, None

        if w and h and (w != current_scr_w or h != current_scr_h):
            scale_x = current_scr_w / w
            scale_y = current_scr_h / h
            scale = min(scale_x, scale_y)
            save_h, save_w = saved_img.shape[:2]
            new_w = max(1, round(save_w * scale))
            new_h = max(1, round(save_h * scale))
            resized_saved = cv2.resize(saved_img, (new_w, new_h))
        else:
            scale = 1
            resized_saved = saved_img

        logger.debug(
            f"loaded {char_id} resized width {current_scr_w} / "
            f"original_width:{w}, scale_x:{scale}"
        )
        return char_id, feature_id, resized_saved

    def match_feature(self, task: "BaseCombatTask", new_image_mat, threshold=0.6, target_char=None):
        """比对新截图与所有数据库内特征图，返回(是/否匹配, 匹配到的角色名, 相似度)"""
        current_scr_h, current_scr_w = task.height, task.width

        with self._data_lock:
            character_snapshot = {}
            for char_id, char_data in self.db["characters"].items():
                if not isinstance(char_data, dict):
                    continue
                character_snapshot[char_id] = list(char_data.get("feature_ids", []))
            current_fids = set()
            for feature_ids in character_snapshot.values():
                current_fids.update(feature_ids)

            need_rebuild = (
                self._cache_scr_w != current_scr_w
                or self._cache_scr_h != current_scr_h
                or self._cache_fids != current_fids
            )
            if need_rebuild:
                self._feature_cache.clear()
                self._cache_scr_w = current_scr_w
                self._cache_scr_h = current_scr_h
                self._cache_fids = current_fids

        if need_rebuild:
            rebuilt_cache = {}
            load_jobs = []
            for char_id, feature_ids in character_snapshot.items():
                rebuilt_cache[char_id] = {}
                for fid in feature_ids:
                    load_jobs.append((char_id, fid))

            worker_count = min(8, max(1, len(load_jobs)))
            if worker_count == 1:
                results = [
                    self._load_resized_feature(char_id, fid, current_scr_w, current_scr_h)
                    for char_id, fid in load_jobs
                ]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    results = executor.map(
                        lambda job: self._load_resized_feature(
                            job[0], job[1], current_scr_w, current_scr_h
                        ),
                        load_jobs,
                    )

            for char_id, fid, resized_saved in results:
                if resized_saved is not None:
                    rebuilt_cache[char_id][fid] = resized_saved

            with self._data_lock:
                self._feature_cache = rebuilt_cache
                box = task.get_box_by_name(Labels.box_char_1)
                self._cache_mask = (
                    create_ellipse_mask(box.width, box.height, box.width * 0.4, box.height * 0.4)
                    if box
                    else None
                )

        with self._data_lock:
            cache_snapshot = {
                char_id: dict(features) for char_id, features in self._feature_cache.items()
            }

        best_match_char_id = None
        best_similarity = 0.0

        for char_id, cached_features in cache_snapshot.items():
            if target_char and char_id != target_char:
                continue
            for fid, cached_mat in cached_features.items():
                mask = None
                if self._cache_mask is not None:
                    if cached_mat.shape[0:2] == self._cache_mask.shape[0:2]:
                        mask = self._cache_mask
                    else:
                        mask = cv2.resize(
                            self._cache_mask,
                            (cached_mat.shape[1], cached_mat.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                if (
                    cached_mat.shape[0] > new_image_mat.shape[0]
                    or cached_mat.shape[1] > new_image_mat.shape[1]
                ):
                    ch = min(cached_mat.shape[0], new_image_mat.shape[0])
                    cw = min(cached_mat.shape[1], new_image_mat.shape[1])
                    cached_mat = cached_mat[:ch, :cw]
                    if mask is not None:
                        mask = mask[:ch, :cw]

                margin = 2
                if cached_mat.shape[0] > margin * 2 and cached_mat.shape[1] > margin * 2:
                    cached_mat = cached_mat[margin:-margin, margin:-margin]
                    if mask is not None:
                        mask = mask[margin:-margin, margin:-margin]

                res = cv2.matchTemplate(new_image_mat, cached_mat, cv2.TM_CCOEFF_NORMED, mask=mask)
                res[np.isinf(res)] = 0
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_similarity:
                    best_similarity = max_val
                    best_match_char_id = char_id

        if best_similarity >= threshold:
            return True, best_match_char_id, best_similarity
        return False, None, best_similarity

    def get_all_characters(self):
        """获取所有角色数据"""
        with self._data_lock:
            characters = {}
            for char_id, char_data in self.db["characters"].items():
                if not isinstance(char_data, dict):
                    continue
                out = dict(char_data)
                char_name = self._character_name_from_record(char_id, char_data)
                combo_id = self._as_text(out.get("combo_id", ""))
                out.pop("name", None)
                out["char_id"] = char_id
                out["char_name"] = char_name
                out["combo_id"] = combo_id
                out["combo_name"] = self.get_combo_name(combo_id)
                characters[char_id] = out
            return characters

    def get_character_combo_id_by_id(self, char_id: str) -> str:
        info = self.get_character_info_by_id(char_id)
        return info["combo_id"] if info else ""

    def get_character_combo_name_by_id(self, char_id: str) -> str:
        return self.get_combo_name(self.get_character_combo_id_by_id(char_id))

    def get_character_info_by_id(self, char_id: str) -> dict | None:
        with self._data_lock:
            char_info = self.db["characters"].get(char_id, None)
            if isinstance(char_info, dict):
                combo_id = self._as_text(char_info.get("combo_id", ""))
                out = dict(char_info)
                char_name = self._character_name_from_record(char_id, char_info)
                out.pop("name", None)
                out["char_id"] = char_id
                out["char_name"] = char_name
                out["combo_id"] = combo_id
                out["combo_name"] = self.get_combo_name(combo_id)
                return out
            return None

    def get_fixed_team(self):
        with self._data_lock:
            fixed_team = self._normalize_fixed_team_config(self.db.get("fixed_team"))
            return {
                "enabled": fixed_team["enabled"],
                "slots": [dict(slot) for slot in fixed_team["slots"]],
            }

    def set_fixed_team(self, enabled: bool, slots):
        with self._data_lock:
            self.db["fixed_team"] = self._normalize_fixed_team_config(
                {
                    "enabled": enabled,
                    "slots": slots,
                }
            )
            self.save_db()

    def clear_fixed_team(self):
        with self._data_lock:
            self.db["fixed_team"] = self._default_fixed_team()
            self.save_db()


def create_ellipse_mask(w, h, rx, ry):
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w // 2), int(h // 2))
    axes = (int(rx), int(ry))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    return mask


def show_masked_template(cached_mat, _cache_mask):
    h, w = cached_mat.shape[:2]

    if len(_cache_mask.shape) == 3:
        mask = _cache_mask[:, :, 0]
    else:
        mask = _cache_mask.copy()

    if mask.shape != (h, w):
        print(f"警告：尺寸不匹配！Mat: {h}x{w}, Mask: {mask.shape}。正在强制 resize...")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = mask.astype(np.uint8)

    result = cv2.bitwise_and(cached_mat, cached_mat, mask=mask)
    result = cv2.resize(result, (w * 5, h * 5), interpolation=cv2.INTER_NEAREST)
    unmasked = cv2.resize(cached_mat, (w * 5, h * 5), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Masked Result", result)
    cv2.imshow("unMasked Result", unmasked)
    cv2.waitKey(0)
