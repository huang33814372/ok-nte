import re
import time
from threading import Event

from ok import TaskDisabledException
from qfluentwidgets import FluentIcon

from src import text_white_color
from src.combat.BaseCombatTask import BaseCombatTask
from src.heist_path.HeistPathA import HeistPathA
from src.Labels import Labels
from src.tasks.NTEOneTimeTask import NTEOneTimeTask
from src.utils import game_filters as gf
from src.utils import image_utils as iu


class AbortException(Exception):
    pass


class AutoHeistTask(NTEOneTimeTask, BaseCombatTask):
    CONF_LOOP_COUNT = "循环次数"
    CONF_PATH = "路径"
    CONF_FIGHTER = "战斗角色"
    CONF_RUNNER = "跑图角色"
    LOCK_PICK_MATCH_THRESHOLD = 0.75

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "自动粉爪大劫案"
        self.icon = FluentIcon.SHOPPING_CART
        self.paths = {
            "路径1": HeistPathA,
        }
        paths_name = list(self.paths.keys())
        self.default_config.update(
            {
                self.CONF_LOOP_COUNT: 0,
                self.CONF_PATH: paths_name[0],
                self.CONF_FIGHTER: ["4", "1"],
                self.CONF_RUNNER: ["3"],
            }
        )
        self.config_description.update(
            {
                self.CONF_LOOP_COUNT: "循环次数, 设置为0则一直运行",
            }
        )

        options = ["1", "2", "3", "4"]
        self.config_type.update(
            {
                self.CONF_PATH: {
                    "type": "drop_down",
                    "options": paths_name,
                },
                self.CONF_FIGHTER: {"type": "multi_selection", "options": options},
                self.CONF_RUNNER: {"type": "multi_selection", "options": options},
            }
        )
        self._scroll_switch = False
        self._scroll_count = 0
        self._scroll_time = 0
        self.dead_fighter = []
        self.quick_pick = Event()
        self._held_keys = set()
        self._spam_key_loop_stop = Event()
        self._spam_key_loop_token = 0

        self.label = 0
        self.error = 0

    def run(self):
        super().run()
        try:
            return self.do_run()
        except TaskDisabledException:
            pass
        except Exception as e:
            self.log_error("自动银行差事出错", e)
            raise

    def do_run(self):
        self.label = ""
        self.error = 0

        count = 0
        earnfcash = 0
        earnpcoin = 0

        total = int(self.config.get(self.CONF_LOOP_COUNT, 1))
        endless = total == 0
        while endless or count < total:
            self.dead_fighter = []
            count += 1
            self.label = f"第 {count} 轮"
            round_text = "∞" if endless else f"{total}"

            self.info_set("轮次", f"{count} / {round_text}")
            self.info_set("失败次数", self.error)
            self.info_add("总方斯获取数", earnfcash)
            self.info_add("总粉爪币获取数", earnpcoin)

            if self.wait_until(self.find_interac, time_out=20, raise_if_not_found=True):
                self.enter_heist()
                if not self.wait_until(self.in_heist, time_out=600):
                    self.heist_error()
                    continue

                try:
                    self.run_path()
                except AbortException as e:
                    self.log_warning(e)
                    self.heist_error()
                    continue

                earnfcash, earnpcoin = self.exit_heist()

            self.next_frame()

    def custom_log(self, message):
        self.log_info(f"{self.label}: " + message)

    def get_earn(self):
        number_re = re.compile(r"(\d+)")
        earnfcash = 0
        earnpcoin = 0

        cash = self.ocr(
            0.359, 0.595, 0.500, 0.642, frame_processor=gf.isolate_text_to_black, name="cash"
        )
        coin = self.ocr(
            0.654, 0.595, 0.789, 0.641, frame_processor=gf.isolate_text_to_black, name="coin"
        )
        if cash:
            match_1 = number_re.search(cash[0].name.replace(",", ""))
            if match_1:
                earnfcash = int(match_1.group(1))

        if coin:
            match_2 = number_re.search(coin[0].name.replace(",", ""))
            if match_2:
                earnpcoin = int(match_2.group(1))

        return earnfcash, earnpcoin

    def heist_error(self):
        self.custom_log("出现异常，将退出粉爪副本")
        self.error += 1
        self.wait_until(
            lambda: self.ocr(0.46, 0.32, 0.54, 0.37, match=re.compile("确认退出")),
            pre_action=lambda: self.send_key("esc", interval=2),
        )
        btn = self.wait_ocr(0.52, 0.63, 0.68, 0.68, match=re.compile("确认"))
        self.wait_until(
            lambda: not self.ocr(0.46, 0.32, 0.54, 0.37, match=re.compile("确认退出")),
            pre_action=lambda: self.operate_click(btn, interval=1),
        )
        self.wait_in_team(time_out=60)
        self.custom_log("已退出粉爪副本")

    # 进入粉爪副本
    def enter_heist(self):
        def in_panel():
            return self.ocr(0.625, 0.483, 0.685, 0.525, match=re.compile("挑战时间"))

        def action():
            self.send_key("f", action_name="enter_heist_f", interval=1)
            if not self.is_in_team():
                self.sleep(0.1)
                self.send_key("space", action_name="enter_heist_space", interval=1)

        self.wait_until(
            in_panel,
            pre_action=action,
            time_out=20,
        )
        self.sleep(0.5)
        self.wait_until(
            lambda: not in_panel(),
            pre_action=lambda: self.operate_click(0.7734, 0.8824, interval=1),
            time_out=20,
        )
        self.sleep(0.5)

    def has_exit_panel(self):
        return self.ocr(0.2602, 0.2639, 0.3520, 0.3257, match=re.compile("安全撤离"))

    # 离开粉爪副本
    def exit_heist(self):

        def in_sum_panel():
            return self.ocr(0.4496, 0.8354, 0.5547, 0.8868, match=re.compile("退出"))

        self.wait_until(
            self.has_exit_panel,
            pre_action=lambda: self.send_key("f", interval=1),
        )
        self.sleep(1)
        earnfcash, earnpcoin = self.get_earn()
        self.wait_until(
            lambda: not self.has_exit_panel(),
            pre_action=lambda: self.operate_click(0.604, 0.701, interval=1),
        )
        self.sleep(1)
        self.wait_until(
            in_sum_panel,
        )
        self.sleep(1)
        self.wait_until(
            lambda: not in_sum_panel(),
            pre_action=lambda: self.operate_click(0.501, 0.864, interval=1),
        )
        self.sleep(1)
        self.wait_in_team(time_out=60)
        self.custom_log("已离开粉爪副本")
        return earnfcash, earnpcoin

    def send_key_down(self, key, after_sleep=0):
        if key == "f":
            if not self.quick_pick.is_set():
                self.quick_pick.ready_at = time.time() + 0.3
            self.quick_pick.set()
            self._scroll_switch = False
            return
        self._held_keys.add(key)
        return super().send_key_down(key, after_sleep)

    def send_key_up(self, key, after_sleep=0):
        if key == "f":
            self._reset_quick_pick()
            return
        try:
            return super().send_key_up(key, after_sleep)
        finally:
            self._held_keys.discard(key)

    def _start_spam_key_loop(self):
        self._spam_key_loop_token += 1
        token = self._spam_key_loop_token
        self._spam_key_loop_stop.clear()
        self.log_info("spam_key_loop start")
        self.submit_periodic_task(0.01, self._spam_key_loop, token)

    def _stop_spam_key_loop(self):
        self._spam_key_loop_stop.set()
        self._reset_quick_pick()

    def _reset_quick_pick(self):
        self.quick_pick.clear()
        if hasattr(self.quick_pick, "ready_at"):
            del self.quick_pick.ready_at

    def _release_held_keys(self):
        held_keys = list(self._held_keys)
        self._held_keys.clear()
        for key in held_keys:
            try:
                super().send_key_up(key)
            except Exception as e:
                self.log_error(f"release held key {key} failed", e)

    def _spam_key_loop(self, token):
        if (
            token != self._spam_key_loop_token
            or self._spam_key_loop_stop.is_set()
            or not self.enabled
            or not self.running
        ):
            self.log_info("spam_key_loop stop")
            return False

        if self.quick_pick.is_set() and time.time() >= getattr(self.quick_pick, "ready_at", 0):
            self.send_key("f", interval=0.25, down_time=0.002)
            self._alternate_scroll(interval=0.25)

    def _alternate_scroll(self, interval=0):
        if time.time() - self._scroll_time >= interval:
            time.sleep(0.01)
            if self._scroll_switch:
                self.scroll(0, 0, 1)
            else:
                self.scroll(0, 0, -1)
            self._scroll_time = time.time()
            self._scroll_count += 1
            if self._scroll_count >= 3:
                self._scroll_count = 0
                self._scroll_switch = not self._scroll_switch

    def run_path(self):
        path_name = self.config.get(self.CONF_PATH)
        path_cls = self.paths.get(path_name, next(iter(self.paths.values())))
        path = path_cls(self)
        self._start_spam_key_loop()
        try:
            return path.run_path()
        finally:
            self._stop_spam_key_loop()
            self._release_held_keys()

    def ensure_in_team(self):
        self.wait_until(self.is_in_team, pre_action=lambda: self.send_key("esc", interval=2))

    def check_current_floor(self, floor=1):
        """检查是否在指定楼层"""
        floor_str = "LG" + str(floor)
        ret = self.wait_ocr(0.04, 0.235, 0.11, 0.275, match=re.compile("LG.*"), time_out=10)
        if ret and floor_str in ret[0].name:
            return
        raise AbortException(f"not in floor {floor}")

    def in_heist(self):
        ret = self.is_in_team() and self.find_one(Labels.heist_timer)
        return ret

    def switch_to_fighter(self):
        keys = self.config.get(self.CONF_FIGHTER, [])
        keys = keys[::-1]
        set_dead = set(self.dead_fighter)
        keys = [item for item in keys if item not in set_dead]
        _key = None
        for key in keys:
            self.send_key(key)
            if self.wait_until(lambda: not self.is_in_team(), time_out=0.6, settle_time=0.2):
                self.log_info(f"char {key} is dead")
                self.ensure_in_team()
            else:
                _key = key
                break
        else:
            raise AbortException(f"fighter {keys} dead or empty")
        return _key

    def switch_to_runner(self):
        keys = self.config.get(self.CONF_RUNNER, [])
        for key in keys:
            self.send_key(key)
            if self.wait_until(lambda: not self.is_in_team(), time_out=0.6, settle_time=0.2):
                self.log_info(f"char {key} is dead")
                self.ensure_in_team()
            else:
                break
        else:
            raise AbortException(f"runner {keys} dead or empty")

    def jump_combat_once(self):
        _key = self.switch_to_fighter()
        self.wait_until(self.has_health_bar)
        deadline = time.time() + 60
        settle = -1
        while time.time() < deadline:
            if settle < 0:
                self.send_key("space")
                self.sleep(0.25)
                self.click()
                self.sleep(0.4)
                if not self.is_in_team():
                    self.log_info(f"fighter {_key} dead, try next")
                    self.dead_fighter.append(_key)
                    self.ensure_in_team()
                    _key = self.switch_to_fighter()
                self.send_key(_key)
                self.next_frame()
            else:
                self.sleep(0.1)
            if not self._find_red_health_bar(10):
                if settle < 0:
                    settle = time.time()
                if time.time() - settle > 2:
                    break
            else:
                settle = -1
        else:
            raise AbortException("timeout for combat_once")
        self.switch_to_runner()

    def wait_send_interac(self, direction=None, key_up_sleep=0.7, is_lock=False, time_out=10):
        ret = self.wait_until(self.find_interac, time_out=time_out)
        if direction is not None:
            self.send_key_up(direction)
        if not ret:
            return False
        self.sleep(key_up_sleep)
        self.wait_until(
            lambda: not self.find_interac(), pre_action=lambda: self.send_key("f", interval=1)
        )
        if is_lock:
            self.wait_until(self.find_lock_pick, time_out=2)
            self.wait_until(lambda: not self.find_lock_pick(), settle_time=0.5)
            return not self.find_interac()
        return True

    def walk_and_loot_safe(self, direction=None, time_out=10, hold=False):
        deadline = time.time() + time_out
        if direction is not None:
            self.send_key_down(direction)
        while time.time() < deadline:
            if self.find_lock_pick():
                lock_pick = time.time()
                if direction is not None:
                    self.send_key_up(direction)
                self.wait_until(lambda: not self.find_lock_pick(), settle_time=0.5)
                self.sleep(0.50)
                deadline += time.time() - lock_pick + 0.2
                if direction is not None:
                    self.send_key_down(direction)
            self.next_frame()
        if direction is not None and not hold:
            self.send_key_up(direction)

    def find_lock_pick(self):
        feature = self.get_feature_by_name(Labels.heist_lock_pick).mat
        box = self.get_box_by_name(Labels.heist_lock_pick).scale(1.5)
        self.draw_boxes(boxes=box, color="blue")
        cropped = box.crop_frame(self.frame)
        cropped = iu.create_color_mask(cropped, text_white_color)
        # iu.show_images([feature, cropped], ["feature", "cropped"])
        res, _ = self._find_rotated_template(
            feature,
            cropped,
            threshold=self.LOCK_PICK_MATCH_THRESHOLD,
            cache_key=Labels.heist_lock_pick,
        )
        return len(res) >= 1

    def is_exit_open(self, direction=None):
        if self.wait_until(self.find_interac):
            if direction is not None:
                self.send_key_up(direction)
                self.sleep(0.40)
            ret = self.wait_until(
                self.has_exit_panel, pre_action=lambda: self.send_key("f", interval=1), time_out=2.6
            )
            if ret:
                self.ensure_in_team()
            return ret
        else:
            raise AbortException("not found exit interaction")

    def walk_until_exit(self, direction=None, time_out=10):
        if direction is not None:
            self.send_key_down(direction)
        self.wait_until(
            self.has_exit_panel, pre_action=self.send_key("f", interval=0.25), time_out=time_out
        )
        if direction is not None:
            self.send_key_up(direction)
        self.exit_heist()
