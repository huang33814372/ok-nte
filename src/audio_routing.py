import atexit
import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ok import ConfigOption, og
from ok.gui.Communicate import communicate
from ok.util.file import get_relative_path
from ok.util.logger import Logger

from src import GAME_EXE

logger = Logger.get_logger(__name__)

SOUND_VOLUME_VIEW_URL = "https://www.nirsoft.net/utils/sound_volume_view.html"
CONFIG_NAME = "Background Audio Routing"
CONF_ENABLE = "Enable Background Audio Routing"
CONF_SOUND_VOLUME_VIEW_PATH = "SoundVolumeView Path"
CONF_BACKGROUND_DEVICE = "Background Output Device"
CONF_OPEN_DOWNLOAD_PAGE = "Open SoundVolumeView Download Page"
CONF_DEVICE_OPTIONS = "_SoundVolumeView Output Devices"
DEFAULT_RENDER_DEVICE = "DefaultRenderDevice"
DEFAULT_DEVICE_OPTIONS = [DEFAULT_RENDER_DEVICE]
_COMMAND_TIMEOUT_SECONDS = 5
_WINDOW_ROUTE_CHECK_INTERVAL_SECONDS = 2
_SOUND_ITEM_COLUMNS = (
    "Name,Command-Line Friendly ID,Item ID,Type,Direction,Device Name,Device,"
    "Default Device,Default Render Device,Output Device,Device State,Process ID,Window Title"
)
_APP_DEVICE_COLUMNS = (
    "Device Name",
    "Device",
    "Default Device",
    "Default Render Device",
    "Output Device",
)
_COMMAND_ID_KEYS = (
    "Command-Line Friendly ID",
    "Command-LineFriendlyID",
    "CommandLineFriendlyID",
)


def create_background_audio_routing_config_option() -> ConfigOption:
    device_options = _initial_device_options()
    connect_background_audio_router()
    return ConfigOption(
        CONFIG_NAME,
        {
            CONF_ENABLE: False,
            CONF_SOUND_VOLUME_VIEW_PATH: "",
            CONF_BACKGROUND_DEVICE: device_options[0],
            CONF_OPEN_DOWNLOAD_PAGE: CONF_OPEN_DOWNLOAD_PAGE,
            CONF_DEVICE_OPTIONS: device_options,
        },
        description=(
            "Optionally route the game to a selected Windows output device while it is in "
            "the background. SoundVolumeView is not bundled; select your own downloaded copy."
        ),
        config_description={
            CONF_ENABLE: "Switch game audio output when the game window leaves the foreground",
            CONF_SOUND_VOLUME_VIEW_PATH: "Select SoundVolumeView.exe downloaded from NirSoft",
            CONF_BACKGROUND_DEVICE: "Output device used while the game is in the background",
            CONF_OPEN_DOWNLOAD_PAGE: "Open the official NirSoft SoundVolumeView page",
        },
        config_type={
            CONF_SOUND_VOLUME_VIEW_PATH: {
                "type": "file_selector",
                "filter": (
                    "SoundVolumeView.exe (SoundVolumeView.exe);;"
                    "Executable Files (*.exe);;All Files (*)"
                ),
                "dialog_title": "Select SoundVolumeView.exe",
            },
            CONF_BACKGROUND_DEVICE: {
                "type": "drop_down",
                "options": device_options,
            },
            CONF_OPEN_DOWNLOAD_PAGE: {
                "type": "button",
                "text": "Open Download Page",
                "callback": open_sound_volume_view_download_page,
            },
        },
        validator=_background_audio_routing_validator(device_options),
    )


def open_sound_volume_view_download_page(*_args, **_kwargs) -> None:
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl(SOUND_VOLUME_VIEW_URL))
    except Exception as exc:
        logger.error("failed to open SoundVolumeView download page", exc)
        _alert_error("Failed to open SoundVolumeView download page")


def discover_output_devices(exe_path: str) -> list[str]:
    return parse_sound_volume_view_devices(_export_sound_items(exe_path))


def discover_app_output_device(exe_path: str, process_name: str = GAME_EXE) -> str:
    return parse_app_output_device(
        _export_sound_items(exe_path),
        process_name=process_name,
    )


def parse_app_output_device(data: Any, process_name: str = GAME_EXE) -> str:
    device_aliases = _render_device_aliases(data)
    for record in _iter_records(data):
        if not _is_application_record(record, process_name):
            continue
        for key in _APP_DEVICE_COLUMNS:
            candidate = _first_text(record, key)
            if not candidate:
                continue
            if _is_valid_output_device_option(candidate):
                return candidate
            device_id = device_aliases.get(candidate.casefold())
            if device_id:
                return device_id
    return DEFAULT_RENDER_DEVICE


def _export_sound_items(exe_path: str):
    if not exe_path:
        raise RuntimeError("Please select SoundVolumeView.exe first")
    if not _is_sound_volume_view_path(exe_path):
        raise RuntimeError("Please select a valid SoundVolumeView.exe file")

    fd, export_path = tempfile.mkstemp(prefix="nte_sound_devices_", suffix=".json")
    os.close(fd)
    try:
        command = [
            exe_path,
            "/SaveFileEncoding",
            "3",
            "/sjson",
            export_path,
            "/Columns",
            _SOUND_ITEM_COLUMNS,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SoundVolumeView failed to export devices, exit code {result.returncode}"
            )
        return _read_json(export_path)
    finally:
        try:
            os.remove(export_path)
        except OSError:
            pass


def parse_sound_volume_view_devices(data: Any) -> list[str]:
    devices = list(DEFAULT_DEVICE_OPTIONS)
    for record in _iter_records(data):
        device_id = _first_text(
            record,
            *_COMMAND_ID_KEYS,
            "Name",
        )
        if not device_id or not _is_render_device(record, device_id):
            continue
        devices.append(device_id)
    return _dedupe_devices(devices)


def audio_route_command(device: str, process_name: str = GAME_EXE) -> list[str]:
    return ["/SetAppDefault", device, "all", process_name]


def connect_background_audio_router() -> None:
    _router.connect_window_signal()


def restore_background_audio_router() -> None:
    _router.restore_on_exit()


def route_background_audio_for_current_window() -> None:
    _router.route_current_window_state()


@dataclass(frozen=True)
class _RouteRequest:
    device: str
    capture_original: bool


class _BackgroundAudioRouter:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending_route: _RouteRequest | None = None
        self._requested_device: str | None = None
        self._original_device: str | None = None
        self._restore_exe_path: str | None = None
        self._restore_needed = False
        self._worker: threading.Thread | None = None
        self._connected = False
        self._bound_exit_event = None
        self._last_visible: bool | None = None
        self.last_mute_check = 0

    def on_window(self, visible: bool, *_args) -> None:
        now = time.time()
        visible_changed = visible != self._last_visible
        recently_checked = now - self.last_mute_check <= _WINDOW_ROUTE_CHECK_INTERVAL_SECONDS
        if not visible_changed and recently_checked:
            return
        self._last_visible = visible
        self.last_mute_check = now
        self.request_route(visible)

    def connect_window_signal(self) -> None:
        self._bind_exit_event()
        with self._lock:
            if self._connected:
                return
            communicate.window.connect(self.on_window)
            self._connected = True

    def request_route(self, visible: bool) -> None:
        self._request_route(visible)

    def route_current_window_state(self) -> None:
        self._bind_exit_event()
        visible = self._current_window_visible()
        if visible is not None:
            self._request_route(visible, enabled=True)

    def _request_route(self, visible: bool, enabled: bool | None = None) -> None:
        self._bind_exit_event()
        config = _routing_config()
        if config is None or not (config.get(CONF_ENABLE, False) if enabled is None else enabled):
            return
        exe_path = config.get(CONF_SOUND_VOLUME_VIEW_PATH, "")
        if not _is_sound_volume_view_path(exe_path):
            logger.warning(
                "background audio routing skipped: SoundVolumeView.exe is not configured"
            )
            return

        device = self._route_device(visible, config)
        if device is None:
            return
        if not device:
            logger.warning("background audio routing skipped: target output device is empty")
            return

        route = _RouteRequest(device=device, capture_original=not visible)
        with self._lock:
            if device == self._requested_device or route == self._pending_route:
                return
            self._pending_route = route
            self._restore_exe_path = exe_path
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._run_pending_routes,
                args=(exe_path,),
                name="background_audio_router",
                daemon=True,
            )
            self._worker.start()

    def _bind_exit_event(self) -> None:
        exit_event = _ok_exit_event()
        if exit_event is None:
            return
        with self._lock:
            if self._bound_exit_event is exit_event:
                return
            exit_event.bind_stop(self)
            self._bound_exit_event = exit_event

    def stop(self) -> None:
        self.restore_on_exit()

    def _current_window_visible(self) -> bool | None:
        with self._lock:
            if self._last_visible is not None:
                return self._last_visible
        hwnd_window = getattr(getattr(og, "device_manager", None), "hwnd_window", None)
        visible = getattr(hwnd_window, "visible", None)
        return visible if isinstance(visible, bool) else None

    def _run_pending_routes(self, exe_path: str) -> None:
        while True:
            with self._lock:
                route = self._pending_route
                self._pending_route = None
                if route is None:
                    self._worker = None
                    return
            if route.capture_original:
                self._ensure_original_device(exe_path)
            device = route.device
            routed = self._apply_route(exe_path, device)
            if routed:
                with self._lock:
                    self._requested_device = device
                    original_device = self._original_device or DEFAULT_RENDER_DEVICE
                    self._restore_needed = device != original_device

    def _ensure_original_device(self, exe_path: str) -> None:
        with self._lock:
            if self._original_device is not None:
                return
        try:
            original_device = discover_app_output_device(exe_path)
        except Exception as exc:
            logger.warning(f"failed to capture original game audio output device: {exc}")
            original_device = DEFAULT_RENDER_DEVICE
        with self._lock:
            if self._original_device is None:
                self._original_device = original_device

    def _route_device(self, visible: bool, config) -> str | None:
        if not visible:
            return config.get(CONF_BACKGROUND_DEVICE)
        with self._lock:
            return self._original_device

    def _apply_route(self, exe_path: str, device: str) -> bool:
        command = [exe_path, *audio_route_command(device)]
        logger.info(
            f"route game audio output: tool={Path(exe_path).name} "
            f"device={device} process={GAME_EXE}"
        )
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_COMMAND_TIMEOUT_SECONDS,
                check=False,
                shell=False,
            )
        except Exception as exc:
            logger.error("failed to route game audio with SoundVolumeView", exc)
            return False
        if result.returncode != 0:
            logger.warning(
                f"SoundVolumeView audio route failed with exit code {result.returncode}"
            )
            return False
        return True

    def restore_on_exit(self) -> None:
        with self._lock:
            worker = self._worker
            self._pending_route = None
        if (
            worker is not None
            and worker is not threading.current_thread()
            and worker.is_alive()
        ):
            worker.join(timeout=_COMMAND_TIMEOUT_SECONDS + 0.5)

        with self._lock:
            exe_path = self._restore_exe_path or _configured_sound_volume_view_path()
            restore_needed = self._restore_needed
            restore_device = self._original_device or DEFAULT_RENDER_DEVICE
        if not restore_needed or not _is_sound_volume_view_path(exe_path):
            return
        logger.info(f"restore game audio output on exit: device={restore_device}")
        if self._apply_route(exe_path, restore_device):
            with self._lock:
                self._requested_device = restore_device
                self._restore_needed = False


def _routing_config():
    global_config = getattr(og, "global_config", None)
    if global_config is None:
        return None
    try:
        return global_config.get_config(CONFIG_NAME)
    except Exception as exc:
        logger.debug(f"background audio routing config unavailable: {exc}")
        return None


def _ok_exit_event():
    exit_event = getattr(og, "exit_event", None)
    if exit_event is not None:
        return exit_event
    ok_instance = getattr(og, "ok", None)
    return getattr(ok_instance, "exit_event", None)


def _configured_sound_volume_view_path() -> str:
    config = _routing_config() or _load_saved_config()
    value = config.get(CONF_SOUND_VOLUME_VIEW_PATH, "")
    return value if isinstance(value, str) else ""


def _initial_device_options() -> list[str]:
    saved_config = _load_saved_config()
    exe_path = saved_config.get(CONF_SOUND_VOLUME_VIEW_PATH, "")
    if _is_sound_volume_view_path(exe_path):
        try:
            return discover_output_devices(exe_path)
        except Exception as exc:
            logger.warning(f"failed to initialize SoundVolumeView output devices: {exc}")
    return _cached_device_options(saved_config)


def _background_audio_routing_validator(device_options: list[str]):
    def validator(key, value):
        if key == CONF_ENABLE:
            if value:
                route_background_audio_for_current_window()
            else:
                restore_background_audio_router()
        if key == CONF_BACKGROUND_DEVICE and value not in device_options:
            return False, "Selected background output device is unavailable"
        if key == CONF_DEVICE_OPTIONS:
            return True, None
        if key == CONF_SOUND_VOLUME_VIEW_PATH:
            if value and not _is_sound_volume_view_path(value):
                return False, "Please select SoundVolumeView.exe"
            if value:
                _refresh_device_options_from_path(value, device_options)
        return True, None

    return validator


def _refresh_device_options_from_path(exe_path: str, device_options: list[str]) -> None:
    try:
        devices = discover_output_devices(exe_path)
    except Exception as exc:
        logger.error("failed to refresh SoundVolumeView output devices", exc)
        _alert_error(str(exc))
        return

    device_options[:] = devices
    _apply_device_options(devices)
    _refresh_global_config_ui()


def _apply_device_options(devices: list[str]) -> None:
    config = _routing_config()
    if config is not None:
        config[CONF_DEVICE_OPTIONS] = devices
        if config.get(CONF_BACKGROUND_DEVICE) not in devices:
            config[CONF_BACKGROUND_DEVICE] = devices[0]

    global_config = getattr(og, "global_config", None)
    option = getattr(global_config, "config_options", {}).get(CONFIG_NAME)
    if option and option.config_type:
        option.config_type[CONF_BACKGROUND_DEVICE]["options"] = devices


def _refresh_global_config_ui() -> None:
    main_window = _get_main_window()
    if main_window is None:
        return

    config = _routing_config()
    setting_tab = getattr(main_window, "setting_tab", None)
    for config_card in getattr(setting_tab, "config_groups", []) or []:
        if getattr(config_card, "config", None) is config:
            _refresh_device_dropdown_widget(config_card)
            config_card.update_config()
            return


def _refresh_device_dropdown_widget(config_widget) -> None:
    widget = getattr(config_widget, "config_widget_by_key", {}).get(CONF_BACKGROUND_DEVICE)
    combo_box = getattr(widget, "combo_box", None)
    if combo_box is None:
        return

    config = _routing_config()
    if config is None:
        return
    devices = config.get(CONF_DEVICE_OPTIONS, [])
    _replace_combo_options(widget, combo_box, devices, config.get(CONF_BACKGROUND_DEVICE))


def _replace_combo_options(widget, combo_box, devices: list[str], current: str) -> None:
    from PySide6.QtGui import QFontMetrics

    combo_box.blockSignals(True)
    try:
        widget.tr_dict = {}
        widget.tr_options = []
        combo_box.clear()
        for device in devices:
            translated = og.app.tr(device)
            widget.tr_options.append(translated)
            widget.tr_dict[translated] = device
        combo_box.addItems(widget.tr_options)
        selected = devices.index(current) if current in devices else 0
        combo_box.setCurrentIndex(selected)
        metrics = QFontMetrics(combo_box.font())
        max_width = max(
            (metrics.horizontalAdvance(option) for option in widget.tr_options),
            default=0,
        )
        combo_box.setFixedWidth(max_width + 50)
    finally:
        combo_box.blockSignals(False)


def _get_main_window():
    main_window = getattr(og, "main_window", None)
    if main_window is not None:
        return main_window

    app = getattr(og, "app", None)
    if app is None:
        return None
    return getattr(app, "main_window", None)


def _load_saved_config() -> dict[str, Any]:
    path = get_relative_path("configs", f"{CONFIG_NAME}.json")
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return config if isinstance(config, dict) else {}


def _cached_device_options(config: dict[str, Any] | None = None) -> list[str]:
    if config is None:
        config = _load_saved_config()
    options = config.get(CONF_DEVICE_OPTIONS)
    if not isinstance(options, list):
        options = []
    current = config.get(CONF_BACKGROUND_DEVICE)
    candidates = _dedupe_devices([*DEFAULT_DEVICE_OPTIONS, *options, current])
    return [device for device in candidates if _is_valid_output_device_option(device)]


def _is_sound_volume_view_path(exe_path: str) -> bool:
    if not exe_path:
        return False
    path = Path(exe_path)
    return path.is_file() and path.name.lower() == "soundvolumeview.exe"


def _read_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def _iter_records(data: Any):
    if isinstance(data, list):
        yield from (item for item in data if isinstance(item, dict))
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))


def _is_render_device(record: dict[str, Any], device_id: str) -> bool:
    item_type = _first_text(record, "Type").casefold()
    direction = _first_text(record, "Direction").casefold()
    record_text = " ".join(str(value) for value in record.values()).lower()
    device_id_lower = device_id.lower()
    if "\\subunit\\" in device_id_lower or " subunit" in record_text:
        return False
    if "\\capture" in device_id_lower or direction == "capture":
        return False
    if item_type and item_type != "device":
        return False
    if "application" in record_text or device_id_lower.endswith(".exe"):
        return False
    if "\\device\\" in device_id_lower and "\\render" in device_id_lower:
        return True
    return item_type == "device" and direction == "render"


def _render_device_aliases(data: Any) -> dict[str, str]:
    aliases = {}
    for record in _iter_records(data):
        device_id = _first_text(
            record,
            *_COMMAND_ID_KEYS,
            "Name",
        )
        if not device_id or not _is_render_device(record, device_id):
            continue
        for key in (
            *_COMMAND_ID_KEYS,
            "Device Name",
            "Name",
            "Item ID",
        ):
            alias = _first_text(record, key)
            if alias:
                aliases.setdefault(alias.casefold(), device_id)
    return aliases


def _is_application_record(record: dict[str, Any], process_name: str) -> bool:
    process_key = process_name.casefold()
    item_type = _first_text(record, "Type").casefold()
    if item_type and item_type != "application":
        return False

    for key in (
        "Name",
        *_COMMAND_ID_KEYS,
        "Item ID",
    ):
        value = _first_text(record, key).casefold()
        if value == process_key or value.endswith(f"\\{process_key}"):
            return True
    return False


def _is_valid_output_device_option(device: str) -> bool:
    if device == DEFAULT_RENDER_DEVICE:
        return True
    device_lower = device.lower()
    if "\\subunit\\" in device_lower or "\\capture" in device_lower:
        return False
    if device_lower.endswith(".exe"):
        return False
    return "\\device\\" in device_lower and "\\render" in device_lower


def _first_text(record: dict[str, Any], *keys: str) -> str:
    normalized = {
        key.lower().replace(" ", "").replace("-", ""): value for key, value in record.items()
    }
    for key in keys:
        value = normalized.get(key.lower().replace(" ", "").replace("-", ""))
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _dedupe_devices(devices: list[str]) -> list[str]:
    result = []
    seen = set()
    for device in devices:
        if not isinstance(device, str) or not device.strip():
            continue
        device = device.strip()
        key = device.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(device)
    return result or list(DEFAULT_DEVICE_OPTIONS)


def _alert_error(message: str) -> None:
    try:
        from ok.gui.util.Alert import alert_error

        alert_error(message)
    except Exception:
        logger.error(message)


_router = _BackgroundAudioRouter()
atexit.register(restore_background_audio_router)
