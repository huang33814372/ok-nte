import json
import os
import shutil
import unittest
import uuid
from unittest.mock import Mock, patch

from src.char.custom.CustomChar import CustomChar
from src.char.custom.CustomCharManager import DB_SCHEMA_VERSION, CustomCharManager

PREDEFINED_CHARACTER_ID = "char_zero"

class TestCustomCharCore(unittest.TestCase):
    def setUp(self):
        temp_root = os.path.join(os.getcwd(), "tests", ".tmp")
        os.makedirs(temp_root, exist_ok=True)
        self.temp_dir = os.path.join(temp_root, f"case_{uuid.uuid4().hex}")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.db_path = os.path.join(self.temp_dir, "db.json")
        self.features_dir = os.path.join(self.temp_dir, "features")
        os.makedirs(self.features_dir, exist_ok=True)

        self.patchers = [
            patch("src.char.custom.CustomCharManager.CUSTOM_CHARS_DIR", self.temp_dir),
            patch("src.char.custom.CustomCharManager.DB_PATH", self.db_path),
            patch("src.char.custom.CustomCharManager.FEATURES_DIR", self.features_dir),
        ]
        for patcher in self.patchers:
            patcher.start()
        CustomCharManager._instance = None

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        CustomCharManager._instance = None

    def _write_db(self, data):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def test_db_schema_migrates_legacy_combo_name(self):
        legacy = {
            "schema_version": 3,
            "combos": {"combo_old": "skill,wait(0.1)"},
            "characters": {
                "char_legacy": {
                    "combo_name": "combo_old",
                    "feature_ids": [],
                }
            },
            "features": {},
        }
        self._write_db(legacy)

        manager = CustomCharManager()
        self.assertEqual(manager.db["schema_version"], DB_SCHEMA_VERSION)
        combo_id = manager.find_custom_combo_id_by_name("combo_old")
        self.assertTrue(combo_id.startswith("combo_"))
        raw = next(iter(manager.db["characters"].values()))
        self.assertEqual(raw["name"], "char_legacy")
        self.assertEqual(raw["combo_id"], combo_id)
        self.assertNotIn("combo_name", raw)
        self.assertNotIn("combo_ref", raw)

        info = manager.get_character_info_by_id(manager._find_character_id_by_name("char_legacy"))
        self.assertIsNotNone(info)
        self.assertEqual(info["combo_id"], combo_id)
        self.assertEqual(info["combo_name"], "combo_old")
        self.assertNotIn("combo_ref", info)

    def test_db_schema_migrates_legacy_builtin_label(self):
        bootstrap = {
            "schema_version": DB_SCHEMA_VERSION,
            "combos": {},
            "characters": {},
            "features": {},
        }
        self._write_db(bootstrap)
        manager = CustomCharManager()
        legacy_builtin_label = (
            f"{manager.get_builtin_prefix()}{manager.get_combo_name(PREDEFINED_CHARACTER_ID)}"
        )

        legacy = {
            "schema_version": 3,
            "combos": {},
            "characters": {
                "char_builtin": {
                    "combo_name": legacy_builtin_label,
                    "feature_ids": [],
                }
            },
            "features": {},
        }
        self._write_db(legacy)
        CustomCharManager._instance = None

        manager = CustomCharManager()
        info = manager.get_character_info_by_id(manager._find_character_id_by_name("char_builtin"))
        self.assertIsNotNone(info)
        self.assertEqual(info["combo_id"], PREDEFINED_CHARACTER_ID)
        self.assertNotIn("combo_ref", info)

    def test_db_schema_remaps_custom_combo_key_conflicting_with_builtin(self):
        legacy = {
            "schema_version": 3,
            "combos": {
                "builtin:char_zero": "skill,wait(0.1)"
            },
            "characters": {
                "char_conflict": {
                    "combo_name": "builtin:char_zero",
                    "feature_ids": [],
                }
            },
            "features": {},
        }
        self._write_db(legacy)

        manager = CustomCharManager()
        remapped_key = manager.find_custom_combo_id_by_name("builtin:char_zero")

        self.assertNotIn("builtin:char_zero", manager.db["combos"])
        self.assertIn(remapped_key, manager.db["combos"])
        self.assertEqual(manager.get_combo(remapped_key), "skill,wait(0.1)")

        info = manager.get_character_info_by_id(manager._find_character_id_by_name("char_conflict"))
        self.assertIsNotNone(info)
        self.assertEqual(info["combo_id"], remapped_key)
        self.assertNotIn("combo_ref", info)
        self.assertEqual(manager.get_combo(info["combo_id"]), "skill,wait(0.1)")

    def test_validate_combo_syntax_reports_line_and_column(self):
        is_valid, error = CustomChar.validate_combo_syntax("skill,wait(0.5)")
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        is_valid, error = CustomChar.validate_combo_syntax("skill(\nwait(0.5)")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("line", error)
        self.assertIn("column", error)

    def test_validate_combo_rejects_unsupported_and_unknown(self):
        is_valid, error = CustomChar.validate_combo_syntax("wait(**data)")
        self.assertFalse(is_valid)
        self.assertIn("**kwargs", error or "")

        is_valid, error = CustomChar.validate_combo_syntax("not_a_command")
        self.assertFalse(is_valid)
        self.assertIn("unknown command", error or "")

    def test_validate_combo_supports_if_command(self):
        is_valid, error = CustomChar.validate_combo_syntax("if_(ultimate, skill)")
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        is_valid, error = CustomChar.validate_combo_syntax("if_(ultimate, l_click(2))")
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        is_valid, error = CustomChar.validate_combo_syntax("if_(ultimate, skill, wait(0.1))")
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_validate_combo_rejects_invalid_if_usage(self):
        is_valid, error = CustomChar.validate_combo_syntax("if_(wait, skill)")
        self.assertFalse(is_valid)
        self.assertIn("not enabled as if_ condition", error or "")

        is_valid, error = CustomChar.validate_combo_syntax("if_(ultimate)")
        self.assertFalse(is_valid)
        self.assertIn("at least 2", error or "")

        is_valid, error = CustomChar.validate_combo_syntax("if_(ultimate, skill, wait=0.1)")
        self.assertFalse(is_valid)
        self.assertIn("only supports positional", error or "")

    def test_if_runtime_executes_then_only_when_condition_is_true_bool(self):
        char = object.__new__(CustomChar)
        char.logger = Mock()
        state = {"then_count": 0}

        cond_true = ("ultimate", lambda self: True, [], {}, "ultimate")
        then_cmds = [
            ("skill", lambda self: state.__setitem__("then_count", state["then_count"] + 1), [], {}, "skill"),
            ("wait", lambda self: state.__setitem__("then_count", state["then_count"] + 1), [], {}, "wait(0.1)"),
        ]
        result = char._execute_if_command(cond_true, then_cmds)
        self.assertTrue(result)
        self.assertEqual(state["then_count"], 2)

        cond_false = ("ultimate", lambda self: False, [], {}, "ultimate")
        result = char._execute_if_command(cond_false, then_cmds)
        self.assertFalse(result)
        self.assertEqual(state["then_count"], 2)

    def test_if_runtime_treats_non_bool_condition_as_false(self):
        char = object.__new__(CustomChar)
        char.logger = Mock()
        state = {"then_count": 0}

        cond_non_bool = ("ultimate", lambda self: "yes", [], {}, "ultimate")
        then_cmds = [("skill", lambda self: state.__setitem__("then_count", state["then_count"] + 1), [], {}, "skill")]
        result = char._execute_if_command(cond_non_bool, then_cmds)

        self.assertFalse(result)
        self.assertEqual(state["then_count"], 0)
        char.logger.warning.assert_called_once()
        self.assertIn("non-bool", char.logger.warning.call_args[0][0])

    def test_validate_db_removes_missing_feature_assets_and_metadata(self):
        existing_fid = "feat_exists"
        missing_fid = "feat_missing"

        with open(os.path.join(self.features_dir, f"{existing_fid}.png"), "wb") as f:
            f.write(b"ok")

        legacy = {
            "schema_version": DB_SCHEMA_VERSION,
            "combos": {},
            "characters": {
                "char_a": {
                    "combo_id": "",
                    "feature_ids": [existing_fid, missing_fid],
                }
            },
            "features": {
                existing_fid: {"width": 1920, "height": 1080},
                missing_fid: {"width": 1920, "height": 1080},
            },
        }
        self._write_db(legacy)

        manager = CustomCharManager()

        char_info = manager.get_character_info_by_id(manager._find_character_id_by_name("char_a"))
        self.assertIsNotNone(char_info)
        self.assertEqual(char_info["feature_ids"], [existing_fid])
        self.assertIn(existing_fid, manager.db["features"])
        self.assertNotIn(missing_fid, manager.db["features"])

    def test_char_name_is_stripped_and_kept_unique(self):
        manager = CustomCharManager()
        raw_name = "  custom hero  "

        char_id = manager.create_character(raw_name, "")
        duplicate_id = manager.create_character("custom hero", "")
        blank_id = manager.create_character("   ", "")

        names = [c["char_name"] for c in manager.get_all_characters().values()]
        self.assertIn("custom hero", names)
        self.assertNotIn(raw_name, names)
        self.assertEqual(names.count("custom hero"), 1)
        self.assertEqual(duplicate_id, char_id)
        self.assertEqual(blank_id, "")

        id_custom = manager._find_character_id_by_name("custom hero")

        self.assertEqual(id_custom, char_id)
        self.assertEqual(manager.get_character_info_by_id(id_custom)["char_name"], "custom hero")
        self.assertNotIn("   ", names)

    def test_character_info_by_id_has_expected_shape(self):
        manager = CustomCharManager()
        char_id = manager.create_character("fixed shape", "")

        info = manager.get_character_info_by_id(char_id)
        missing = manager.get_character_info_by_id("missing")

        self.assertEqual(info["char_id"], char_id)
        self.assertEqual(info["char_name"], "fixed shape")
        self.assertEqual(info["combo_id"], "")
        self.assertEqual(info["combo_name"], "")
        self.assertEqual(info["feature_ids"], [])
        self.assertNotIn("name", info)

        self.assertIsNone(missing)

    def test_fixed_team_migrates_combo_ref_to_combo_id(self):
        legacy = {
            "schema_version": 4,
            "combos": {},
            "characters": {
                "char_001": {"name": "零", "combo_ref": "builtin:char_zero"}
            },
            "features": {},
            "fixed_team": {
                "enabled": True,
                "slots": [
                    {"char_name": "零", "combo_ref": "builtin:char_zero"},
                ],
            },
        }
        self._write_db(legacy)

        manager = CustomCharManager()
        fixed_team = manager.get_fixed_team()

        self.assertTrue(fixed_team["enabled"])
        char_id = ""
        for cid, cdata in manager.db["characters"].items():
            if cdata["name"] == "零":
                char_id = cid
                break
        self.assertNotEqual(char_id, "")
        self.assertEqual(fixed_team["slots"][0]["char_id"], char_id)
        self.assertEqual(fixed_team["slots"][0]["combo_id"], PREDEFINED_CHARACTER_ID)
        self.assertNotIn("combo_ref", fixed_team["slots"][0])


if __name__ == "__main__":
    unittest.main()
