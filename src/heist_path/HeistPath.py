from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tasks.AutoHeistTask import AutoHeistTask

    _HeistPathTaskProxy = AutoHeistTask
else:

    class _HeistPathTaskProxy:
        pass


class HeistPath(_HeistPathTaskProxy):
    def __init__(self, task: AutoHeistTask):
        self.exit_state = {1: False, 2: False, 3: False, 4: False, }
        self.task = task

    def __getattr__(self, name: str) -> Any:
        return getattr(self.task, name)

    def sleep(self, timeout):
        target = time.perf_counter() + timeout
        if timeout > 0.02:
            self.task.sleep(timeout - 0.02)
        while True:
            if time.perf_counter() >= target:
                break
        return True
