"""Test package setup for unittest discovery runners."""

from __future__ import annotations

import ok as _ok
import ok.gui.util.app as _ok_app_util
import ok.test as _ok_test
from ok.gui.common.config import cfg
from ok.util.handler import ExitEvent
from PySide6.QtWidgets import QApplication

_original_init_app_config = _ok_app_util.init_app_config
_original_init_ok = _ok_test.init_ok
_original_destroy_ok = _ok_test.destroy_ok


def _init_app_config_reusing_qapplication():
    app = QApplication.instance()
    if app is None:
        return _original_init_app_config()
    return app, cfg.get(cfg.language).value


def _reset_ok_runtime_state():
    ExitEvent.queues = set()
    ExitEvent.to_stops = set()
    _ok.OK.exit_event = ExitEvent()
    for attr in (
        "executor",
        "feature_set",
        "device_manager",
        "ocr",
        "overlay_window",
        "screenshot",
        "init_error",
    ):
        setattr(_ok.OK, attr, None)
    for attr in (
        "app",
        "executor",
        "device_manager",
        "handler",
        "my_app",
        "ok",
        "config",
        "task_manager",
        "global_config",
    ):
        setattr(_ok.og, attr, None)


def _init_ok_with_fresh_runtime(config):
    _ok_test.ok = None
    _reset_ok_runtime_state()
    return _original_init_ok(config)


def _destroy_ok_and_clear_singleton():
    try:
        return _original_destroy_ok()
    finally:
        _ok_test.ok = None
        _reset_ok_runtime_state()


_ok_app_util.init_app_config = _init_app_config_reusing_qapplication
_ok_test.init_ok = _init_ok_with_fresh_runtime
_ok_test.destroy_ok = _destroy_ok_and_clear_singleton
