import json
import os
import re
from collections.abc import Callable, Iterable


DB_SCHEMA_VERSION = 5
LEGACY_BUILTIN_PREFIX = "builtin:"
LEGACY_KEY_PATTERN = re.compile(r"\(([^)]+)\)\s*$")


def as_text(value) -> str:
    return "" if value is None else str(value)


def is_blank_text(value) -> bool:
    return as_text(value).strip() == ""


def default_fixed_team():
    return {"enabled": False, "slots": [{"char_id": "", "combo_id": ""} for _ in range(4)]}


def normalize_fixed_team_slot(slot) -> dict:
    slot = slot if isinstance(slot, dict) else {}
    char_id = as_text(slot.get("char_id", "")).strip()
    combo_id = as_text(slot.get("combo_id", "")).strip()
    if is_blank_text(char_id):
        char_id = ""
        combo_id = ""
    return {
        "char_id": char_id,
        "combo_id": combo_id,
    }


def normalize_fixed_team_config(config) -> dict:
    normalized = default_fixed_team()
    if not isinstance(config, dict):
        return normalized

    normalized["enabled"] = bool(config.get("enabled", False))
    raw_slots = config.get("slots", [])
    if isinstance(raw_slots, list):
        for i in range(min(4, len(raw_slots))):
            normalized["slots"][i] = normalize_fixed_team_slot(raw_slots[i])
    return normalized


def default_db():
    return {
        "schema_version": DB_SCHEMA_VERSION,
        "combos": {},
        "characters": {},
        "features": {},
        "fixed_team": default_fixed_team(),
    }


def character_name_from_record(char_id: str, char_data: dict) -> str:
    name = as_text(char_data.get("name", "")).strip()
    if not is_blank_text(name):
        return name
    fallback = as_text(char_id).strip()
    return fallback if not is_blank_text(fallback) else "unnamed"


def unique_name(name: str, used_names: set[str]) -> str:
    name = name.strip()
    if is_blank_text(name):
        name = "unnamed"
    candidate = name
    suffix = 2
    while candidate in used_names:
        candidate = f"{name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def load_db(db_path: str, logger=None) -> dict:
    loaded = default_db()
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                loaded["schema_version"] = data.get("schema_version", 0)
                loaded["combos"] = data.get("combos", loaded["combos"])
                loaded["characters"] = data.get("characters", loaded["characters"])
                loaded["features"] = data.get("features", loaded["features"])
                loaded["fixed_team"] = data.get("fixed_team", loaded["fixed_team"])
        except Exception as e:
            if logger:
                logger.error("Failed to load custom char DB", e)
    return loaded


def save_db(db_path: str, db: dict, logger=None):
    try:
        db["schema_version"] = DB_SCHEMA_VERSION
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        temp_path = db_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, db_path)
    except Exception as e:
        if logger:
            logger.error("Failed to save custom char DB", e)


def validate_db(
    db: dict,
    features_dir: str,
    is_builtin_combo: Callable[[str], bool],
) -> bool:
    modified = False

    if not isinstance(db.get("combos"), dict):
        db["combos"] = {}
        modified = True

    valid_combos = {}
    for combo_id, combo_data in db["combos"].items():
        combo_id = as_text(combo_id).strip()
        if is_blank_text(combo_id) or is_builtin_combo(combo_id):
            modified = True
            continue
        if isinstance(combo_data, dict):
            combo_name = as_text(combo_data.get("name", combo_id)).strip()
            content = as_text(combo_data.get("content", ""))
        else:
            combo_name = combo_id
            content = as_text(combo_data)
            modified = True
        if is_blank_text(combo_name):
            combo_name = combo_id
            modified = True
        valid_combos[combo_id] = {"name": combo_name, "content": content}
    if valid_combos != db["combos"]:
        db["combos"] = valid_combos
        modified = True

    if not isinstance(db.get("characters"), dict):
        db["characters"] = {}
        modified = True

    if not isinstance(db.get("features"), dict):
        db["features"] = {}
        modified = True

    fixed_team = normalize_fixed_team_config(db.get("fixed_team"))
    if fixed_team != db.get("fixed_team"):
        db["fixed_team"] = fixed_team
        modified = True

    used_names = set()
    for char_id, char_data in db["characters"].items():
        if not isinstance(char_data, dict):
            db["characters"][char_id] = {
                "name": unique_name(as_text(char_id), used_names),
                "combo_id": "",
                "feature_ids": [],
            }
            modified = True
            continue

        char_name = unique_name(character_name_from_record(char_id, char_data), used_names)
        if char_data.get("name") != char_name:
            char_data["name"] = char_name
            modified = True

        combo_id = as_text(char_data.get("combo_id", "")).strip()
        if combo_id and not is_builtin_combo(combo_id) and combo_id not in db["combos"]:
            combo_id = ""
            modified = True
        if combo_id != char_data.get("combo_id", ""):
            char_data["combo_id"] = combo_id
            modified = True

        feature_ids = char_data.get("feature_ids", [])
        if not isinstance(feature_ids, list):
            feature_ids = []
            modified = True
        valid_fids = []
        for fid in feature_ids:
            path = os.path.join(features_dir, f"{fid}.png")
            if os.path.exists(path):
                valid_fids.append(fid)
            else:
                modified = True
        char_data["feature_ids"] = valid_fids

    for fid in list(db["features"].keys()):
        path = os.path.join(features_dir, f"{fid}.png")
        if not os.path.exists(path):
            del db["features"][fid]
            modified = True

    return modified


def legacy_builtin_label_to_combo_id(
    value: str,
    get_builtin_prefix: Callable[[], str],
    is_builtin_combo: Callable[[str], bool],
    iter_builtin_combo_items: Callable[[], Iterable[tuple[str, str]]],
) -> str | None:
    if not value:
        return None

    prefixes = [get_builtin_prefix(), "[内置代码] "]
    prefix = next((prefix for prefix in prefixes if value.startswith(prefix)), "")
    if not prefix:
        return None

    label = value.replace(prefix, "", 1).strip()
    match = LEGACY_KEY_PATTERN.search(label)
    if match:
        key = match.group(1).strip()
        if is_builtin_combo(key):
            return key

    if is_builtin_combo(label):
        return label

    matched_ids = [
        combo_id for combo_name, combo_id in iter_builtin_combo_items() if combo_name == label
    ]
    if len(matched_ids) == 1:
        return matched_ids[0]
    return None


def legacy_value_to_combo_id(
    value: str,
    combo_id_remap: dict[str, str],
    get_builtin_prefix: Callable[[], str],
    is_builtin_combo: Callable[[str], bool],
    iter_builtin_combo_items: Callable[[], Iterable[tuple[str, str]]],
) -> str:
    raw_value = as_text(value)
    value = raw_value.strip()
    if not value:
        return ""
    if value in combo_id_remap:
        return combo_id_remap[value]
    if value.startswith(LEGACY_BUILTIN_PREFIX):
        key = value[len(LEGACY_BUILTIN_PREFIX) :].strip()
        if is_builtin_combo(key):
            return key
    if is_builtin_combo(value):
        return value
    legacy_builtin_id = legacy_builtin_label_to_combo_id(
        value,
        get_builtin_prefix,
        is_builtin_combo,
        iter_builtin_combo_items,
    )
    if legacy_builtin_id:
        return legacy_builtin_id
    return value


def migrate_db_schema(
    db: dict,
    is_builtin_combo: Callable[[str], bool],
    get_builtin_prefix: Callable[[], str],
    iter_builtin_combo_items: Callable[[], Iterable[tuple[str, str]]],
    generate_combo_id: Callable[[set[str] | None], str],
) -> tuple[dict, bool]:
    source_schema_version = db.get("schema_version", 0)
    try:
        source_schema_version = int(source_schema_version)
    except (TypeError, ValueError):
        source_schema_version = 0

    if source_schema_version >= DB_SCHEMA_VERSION:
        return db, False

    raw_combos = db.get("combos", {})
    raw_characters = db.get("characters", {})
    raw_features = db.get("features", {})
    raw_fixed_team = db.get("fixed_team", default_fixed_team())
    raw_combos = raw_combos if isinstance(raw_combos, dict) else {}
    raw_characters = raw_characters if isinstance(raw_characters, dict) else {}
    raw_features = raw_features if isinstance(raw_features, dict) else {}

    normalized_combos = {}
    combo_id_remap = {}
    existing_combo_ids = set()
    for old_combo_key, combo_content in raw_combos.items():
        old_combo_key = as_text(old_combo_key)
        if is_blank_text(old_combo_key):
            continue
        combo_id = generate_combo_id(existing_combo_ids)
        existing_combo_ids.add(combo_id)
        combo_id_remap[old_combo_key.strip()] = combo_id
        normalized_combos[combo_id] = {
            "name": old_combo_key.strip(),
            "content": as_text(combo_content),
        }

    normalized_characters = {}
    used_names = set()
    legacy_id_index = 1

    def next_legacy_id():
        nonlocal legacy_id_index
        while True:
            candidate = f"char_{legacy_id_index:04d}"
            legacy_id_index += 1
            if candidate not in normalized_characters:
                return candidate

    for raw_char_id, raw_char_data in raw_characters.items():
        source_data = raw_char_data if isinstance(raw_char_data, dict) else {}
        raw_char_id = as_text(raw_char_id).strip()
        if "name" in source_data:
            char_name = as_text(source_data.get("name", "")).strip()
            char_id = raw_char_id if raw_char_id else next_legacy_id()
        else:
            char_name = raw_char_id
            char_id = next_legacy_id()
        char_name = unique_name(char_name, used_names)

        combo_value = as_text(source_data.get("combo_id", ""))
        if not combo_value:
            combo_value = as_text(source_data.get("combo_ref", ""))
        if not combo_value:
            combo_value = as_text(source_data.get("combo_name", ""))
        combo_id = legacy_value_to_combo_id(
            combo_value,
            combo_id_remap,
            get_builtin_prefix,
            is_builtin_combo,
            iter_builtin_combo_items,
        )
        if combo_id and not is_builtin_combo(combo_id) and combo_id not in normalized_combos:
            combo_id = ""

        feature_ids = source_data.get("feature_ids", [])
        if not isinstance(feature_ids, list):
            feature_ids = []

        while char_id in normalized_characters:
            char_id = next_legacy_id()
        normalized_characters[char_id] = {
            "name": char_name,
            "combo_id": combo_id,
            "feature_ids": feature_ids,
        }

    normalized_fixed_team = default_fixed_team()
    if isinstance(raw_fixed_team, dict):
        normalized_fixed_team["enabled"] = bool(raw_fixed_team.get("enabled", False))
        raw_slots = raw_fixed_team.get("slots", [])
        if isinstance(raw_slots, list):
            for i in range(min(4, len(raw_slots))):
                slot = raw_slots[i] if isinstance(raw_slots[i], dict) else {}
                char_name = as_text(slot.get("char_name", "")).strip()
                combo_value = as_text(slot.get("combo_id", ""))
                if not combo_value:
                    combo_value = as_text(slot.get("combo_ref", ""))
                combo_id = legacy_value_to_combo_id(
                    combo_value,
                    combo_id_remap,
                    get_builtin_prefix,
                    is_builtin_combo,
                    iter_builtin_combo_items,
                )
                char_id = ""
                for cid, cdata in normalized_characters.items():
                    if cdata["name"] == char_name:
                        char_id = cid
                        break
                if is_blank_text(char_id):
                    char_id = ""
                    combo_id = ""
                normalized_fixed_team["slots"][i] = {
                    "char_id": char_id,
                    "combo_id": combo_id,
                }

    return (
        {
            "schema_version": DB_SCHEMA_VERSION,
            "combos": normalized_combos,
            "characters": normalized_characters,
            "features": raw_features,
            "fixed_team": normalized_fixed_team,
        },
        True,
    )
