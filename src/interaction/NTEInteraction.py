import ctypes
import threading
import time
from contextlib import suppress

import win32api
import win32con
import win32gui
from ok import og
from ok.device.intercation import (
    INPUT,
    MOUSEINPUT,
    PostMessageInteraction,
    SendInput,
)
from ok.util.logger import Logger
from win32api import GetCursorPos, SetCursorPos

from src.interaction.keyboard_layout import QwertyPhysicalKeyMapper

logger = Logger.get_logger(__name__)
CHECK_CURSOR_KEY = ["m", "f", "esc"]
for i in range(1, 13):
    CHECK_CURSOR_KEY.append(f"f{i}")


def _cursor_sync_worker(lock, state_box):
    while True:
        with lock:
            state = state_box.get("state")
            if state is None:
                state_box["thread"] = None
                return

        cursor_pos = None
        with suppress(Exception):
            cursor_pos = GetCursorPos()

        can_check_cursor = all((cursor_pos is not None, time.time() < state["deadline"]))
        should_reset = False
        if can_check_cursor:
            assert cursor_pos is not None
            curr_x, curr_y = cursor_pos
            abs_center_x, abs_center_y = state["center"]
            limit_x, limit_y = state["limit"]
            last_x, last_y = state["last_cursor_position"]
            is_in_center_zone = all(
                (abs(curr_x - abs_center_x) <= limit_x, abs(curr_y - abs_center_y) <= limit_y)
            )
            is_far_from_last_position = any(
                (abs(curr_x - last_x) > limit_x, abs(curr_y - last_y) > limit_y)
            )
            should_reset = all((is_in_center_zone, is_far_from_last_position))

            if not is_in_center_zone:
                with lock:
                    if state_box.get("state") is state:
                        state["last_cursor_position"] = cursor_pos

        if all((can_check_cursor, not should_reset)):
            time.sleep(0.01)
            continue

        with lock:
            if state_box.get("state") is not state:
                continue
            state_box["state"] = None
            state_box["thread"] = None
            if should_reset:
                SetCursorPos(state["last_cursor_position"])
        return


class NTEInteraction(PostMessageInteraction):
    _ACTIVATE_REFRESH_INTERVAL = 60 * 60

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_position = None
        self._operating = False
        self._input_lock = threading.RLock()
        self.user32 = ctypes.windll.user32
        self.qwerty_physical_key_mapper = QwertyPhysicalKeyMapper()
        self._disable_key_mapping = 0
        self._activate_require = True
        self._next_try_activate_at = -1
        self._cursor_sync_lock = threading.Lock()
        self._cursor_sync_state = {"state": None, "thread": None}
        self.hwnd_window.visible_monitors.append(self)

    def on_visible(self, visible):
        self._activate_require = not visible

    def send_key(self, key, down_time=0.01):
        with self._input_lock:
            cursor_position = None
            if key in CHECK_CURSOR_KEY:
                with suppress(Exception):
                    cursor_position = GetCursorPos()
            key = self._map_key(key)
            self._disable_key_mapping += 1
            try:
                return super().send_key(key, down_time=down_time)
            finally:
                if cursor_position:
                    self.monitor_and_sync_cursor(cursor_position, timeout=0.3)
                self._disable_key_mapping -= 1

    def send_key_down(self, key, activate=True):
        with self._input_lock:
            key = self._map_key(key)
            return super().send_key_down(key, activate=activate)

    def send_key_up(self, key):
        with self._input_lock:
            key = self._map_key(key)
            return super().send_key_up(key)

    def scroll(self, x, y, scroll_amount):
        with self._input_lock:
            self.try_activate()
            logger.debug(f"scroll {x}, {y}, {scroll_amount}")

            base_hwnd = (
                self.hwnd_window.top_hwnd if self.hwnd_window.top_hwnd else self.hwnd_window.hwnd
            )
            if x > 0 and y > 0:
                top_x, top_y = self.hwnd_window.get_top_window_cords(x, y)
                abs_x, abs_y = win32gui.ClientToScreen(base_hwnd, (int(top_x), int(top_y)))
                self.bg_mouse_pos = (top_x, top_y)
                self._dynamic_target_hwnd = self._target_hwnd_at(abs_x, abs_y, base_hwnd)
                long_position = win32api.MAKELONG(abs_x, abs_y)
            else:
                self._dynamic_target_hwnd = base_hwnd
                long_position = 0

            wparam = win32api.MAKELONG(0, win32con.WHEEL_DELTA * scroll_amount)
            self.post(win32con.WM_MOUSEWHEEL, wparam, long_position)

    def _target_hwnd_at(self, abs_x, abs_y, fallback_hwnd):
        for hwnd_info in getattr(self.hwnd_window, "hwnds", []):
            candidate = hwnd_info[0]
            if not win32gui.IsWindow(candidate):
                continue
            try:
                left = hwnd_info[4]
                top = hwnd_info[5]
                right = left + hwnd_info[2]
                bottom = top + hwnd_info[3]
                if left <= abs_x < right and top <= abs_y < bottom:
                    return candidate
            except Exception:
                continue
        return fallback_hwnd

    def _map_key(self, key):
        if self._disable_key_mapping or not og.global_config.get_config("Game Hotkey Config").get(
            "Use QWERTY Physical Keys", False
        ):
            return key

        return self.qwerty_physical_key_mapper.map_key(key) or key

    def click(self, x=-1, y=-1, move_back=False, name=None, down_time=0.01, move=True, key="left"):
        with self._input_lock:
            self.try_activate()
            if x < 0:
                x, y = round(self.capture.width * 0.5), round(self.capture.height * 0.5)

            should_restore = move and move_back and not self._operating
            if move:
                if should_restore:
                    self.cursor_position = GetCursorPos()
                abs_x, abs_y = self.capture.get_abs_cords(x, y)
                SetCursorPos((abs_x, abs_y))
                time.sleep(0.035)
            click_pos = win32api.MAKELONG(x, y)
            if key == "left":
                btn_down = win32con.WM_LBUTTONDOWN
                btn_mk = win32con.MK_LBUTTON
                btn_up = win32con.WM_LBUTTONUP
            elif key == "middle":
                btn_down = win32con.WM_MBUTTONDOWN
                btn_mk = win32con.MK_MBUTTON
                btn_up = win32con.WM_MBUTTONUP
            else:
                btn_down = win32con.WM_RBUTTONDOWN
                btn_mk = win32con.MK_RBUTTON
                btn_up = win32con.WM_RBUTTONUP
            self.post(btn_down, btn_mk, click_pos)
            time.sleep(down_time)
            self.post(btn_up, 0, click_pos)
            if should_restore:
                self._restore_cursor()

    def operate(self, fun, block=False, restore_cursor=True):
        with self._input_lock:
            result = None
            reset_scene_executor = None

            is_outer_operate = False
            if not self._operating:
                self.cursor_position = GetCursorPos()
                self._operating = True
                is_outer_operate = True

            if block:
                self.block_input()
            try:
                reset_scene_executor = self._enter_reset_scene_without_check()
                result = fun()
            except Exception as e:
                logger.error("operate exception", e)
                raise
            finally:
                self._exit_reset_scene_without_check(reset_scene_executor)
                if is_outer_operate:
                    self._operating = False
                    if restore_cursor:
                        self._restore_cursor()
                if block:
                    self.unblock_input()
            return result

    def _enter_reset_scene_without_check(self):
        device_manager = getattr(self.hwnd_window, "device_manager", None)
        executor = getattr(device_manager, "executor", None)
        if executor is None:
            return None

        if not hasattr(executor, "_nte_operate_reset_scene_depth"):
            executor._nte_operate_reset_scene_depth = 0

        if not hasattr(executor, "_nte_original_reset_scene"):
            executor._nte_original_reset_scene = executor.reset_scene

            def reset_scene(check_enabled=True):
                if executor._nte_operate_reset_scene_depth > 0:
                    check_enabled = False
                return executor._nte_original_reset_scene(check_enabled=check_enabled)

            executor.reset_scene = reset_scene

        executor._nte_operate_reset_scene_depth += 1
        return executor

    def _exit_reset_scene_without_check(self, executor):
        if executor is None:
            return
        executor._nte_operate_reset_scene_depth = max(
            0, executor._nte_operate_reset_scene_depth - 1
        )

    def _restore_cursor(self):
        time.sleep(0.035)
        try:
            SetCursorPos(self.cursor_position)
        except Exception as e:
            logger.error("restore cursor exception", e)

    def block_input(self):
        self.user32.BlockInput(True)

    def unblock_input(self):
        self.user32.BlockInput(False)

    def move_mouse_relative(self, dx, dy):
        """
        Moves the mouse cursor relative to its current position using user32.SendInput.

        Args:
            dx: The number of pixels to move the mouse horizontally.
                (positive for right, negative for left).
            dy: The number of pixels to move the mouse vertically.
                (positive for down, negative for up).
        """

        mi = MOUSEINPUT(dx, dy, 0, 1, 0, None)
        i = INPUT(0, mi)  # type=0 indicates a mouse event
        SendInput(1, ctypes.pointer(i), ctypes.sizeof(INPUT))

    def try_activate(self):
        now = time.monotonic()
        if self._activate_require:
            if not self.hwnd_window.is_foreground():
                super().try_activate()
                self._next_try_activate_at = now + self._ACTIVATE_REFRESH_INTERVAL
            self._activate_require = False
        elif now >= self._next_try_activate_at:
            super().try_activate()
            self._next_try_activate_at = now + self._ACTIVATE_REFRESH_INTERVAL

    def monitor_and_sync_cursor(self, cursor_position=None, timeout=2.0, threshold_ratio=0.05):
        """
        在指定 timeout 时间内异步进行高频监测。
        如果鼠标回到捕获区域中心 5% 范围内且远离目标点，则强制重置。

        :param cursor_position: 监测开始时的鼠标坐标，用于初始化恢复位置 (x, y)
        :param timeout: 监测持续时间（秒）
        :param threshold_ratio: 判定阈值比例，默认 0.05 (5%)
        """
        if self.hwnd_window.is_foreground():
            return

        if cursor_position is None:
            cursor_position = GetCursorPos()

        # --- 1. 预计算固定值（移出循环以提高频率） ---
        # 计算捕获区域的绝对中心点坐标
        c_rel_x, c_rel_y = round(self.capture.width * 0.5), round(self.capture.height * 0.5)
        abs_center_x, abs_center_y = self.capture.get_abs_cords(c_rel_x, c_rel_y)

        # 计算 5% 的阈值
        limit_x = self.capture.width * threshold_ratio
        limit_y = self.capture.height * threshold_ratio

        state = {
            "center": (abs_center_x, abs_center_y),
            "deadline": time.time() + timeout,
            "limit": (limit_x, limit_y),
            "last_cursor_position": tuple(cursor_position),
        }
        with self._cursor_sync_lock:
            self._cursor_sync_state["state"] = state
            thread = self._cursor_sync_state.get("thread")
            if thread is not None and thread.is_alive():
                return

            thread = threading.Thread(
                target=_cursor_sync_worker,
                args=(self._cursor_sync_lock, self._cursor_sync_state),
                daemon=True,
                name="NTECursorSync",
            )
            self._cursor_sync_state["thread"] = thread
            thread.start()
