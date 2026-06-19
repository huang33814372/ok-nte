import time

from qfluentwidgets import FluentIcon

from ok import TaskDisabledException
from src.combat.BaseCombatTask import BaseCombatTask
from src.sound_trigger.SoundCombatContext import SoundCombatContext
from src.tasks.NTEOneTimeTask import NTEOneTimeTask


class WhirlwindTask(NTEOneTimeTask, BaseCombatTask):
    TARGET_NAVIGATION_ANGLE = -154.5
    NAVIGATION_ANGLE_TOLERANCE = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "自动小旋风"
        self.description = "可交互「小旋风」下按开始"
        self.icon = FluentIcon.FLAG

    def run(self):
        super().run()
        try:
            self.do_run()
        except TaskDisabledException:
            pass
        except Exception as e:
            self.log_error("WhirlwindTask Error", e)

    def do_run(self):
        SoundCombatContext().update_task(self, dodge_action=self._dodge_with_skill)
        try:
            while True:
                if not self.is_boss():
                    if not self.find_interac():
                        self.navigate()
                        self.sleep(0.1)
                    self.start_interac()
                self.start_combat()
        finally:
            self._release_navigation_keys()

    def _dodge_with_skill(self):
        self.send_key(self.get_skill_key())

    def start_interac(self):
        self.send_key_down("w")
        self.wait_until(self.find_interac)
        self.send_key_up("w")
        self.wait_until(
            lambda: not self.is_in_team(),
            pre_action=lambda: self.send_key("f", interval=0.25),
            time_out=20,
        )
        self.wait_until(
            self.is_in_team,
            pre_action=lambda: self.operate_click(0.701, 0.542, interval=2),
            settle_time=1,
            time_out=60,
        )

    def start_combat(self):
        if not self.is_boss():
            self.send_key_down("w")
            self.sleep(0.2)
            self.send_key("lshift")
            self.wait_until(self.is_boss)
            self.sleep(1)
            self.send_key_up("w")
        with self.skip_sleep_checks() as skip:
            skip.check_combat = True
            while True:
                self.click()
                self.sleep(0.15)
                self.send_key(self.get_ultimate_key())
                self.sleep(0.15)
                if not self.is_boss() and self.is_in_team():
                    break
        self.sleep(2)
        self.wait_in_team(time_out=60, settle_time=1)
        self.click(key="middle")
        self.sleep(1)

    def navigate(self):
        self.wait_in_team()
        failed_time = 0
        try:
            while True:
                ret = self.check_mini_map_arrow()
                if not ret:
                    if failed_time == 0:
                        failed_time = time.time()
                else:
                    failed_time = 0

                if ret:
                    angle = ret[0].get("angle")
                    if angle is None:
                        if failed_time == 0:
                            failed_time = time.time()
                    else:
                        failed_time = 0

                if failed_time != 0:
                    if time.time() - failed_time > 4:
                        return False
                    self.sleep(0.25)
                    continue

                error = self._normalize_angle(self.TARGET_NAVIGATION_ANGLE - angle)
                if abs(error) <= self.NAVIGATION_ANGLE_TOLERANCE:
                    return True

                self._fine_tune_navigation(self._navigation_side_key(angle), abs(error))
        finally:
            self._release_navigation_keys()

    def _navigation_side_key(self, current_angle):
        error = self._normalize_angle(self.TARGET_NAVIGATION_ANGLE - current_angle)
        return "d" if error < 0 else "a"

    def _fine_tune_navigation(self, side_key, angle_diff):
        if angle_diff > 90:
            self.send_key(side_key)
            self.sleep(0.5)
        else:
            try:
                self.send_key_down(side_key)
                self.send_key("w")
            finally:
                self.send_key_up(side_key)
            self.sleep(0.05)

        self.click(key="middle")
        self.sleep(0.5)
        self.send_key("w")
        self.sleep(0.5)
        self.next_frame()

    def _release_navigation_keys(self):
        self.send_key_up("a")
        self.send_key_up("d")
