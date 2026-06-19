from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Optional

import psutil


def _foreground_pid() -> Optional[int]:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value) or None
    except Exception:
        return None


def _process_name(pid: int) -> Optional[str]:
    try:
        return psutil.Process(pid).name()
    except Exception:
        return None


def name_set(process_name) -> set[str]:
    if isinstance(process_name, (list, tuple, set)):
        names = process_name
    else:
        names = [process_name]
    return {str(name).casefold() for name in names if name}


def resolve_target_pid(process_name) -> Optional[int]:
    targets = name_set(process_name)
    if not targets:
        return None

    foreground = _foreground_pid()
    if foreground and (_process_name(foreground) or "").casefold() in targets:
        return foreground

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if (proc.info.get("name") or "").casefold() in targets:
                return int(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue
    return None


def process_is_alive(pid: int, process_name) -> bool:
    return (_process_name(pid) or "").casefold() in name_set(process_name)
