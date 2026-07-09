import time

from src.char.BaseChar import BaseChar
from src.combat.planner import (
    ActionSlot,
    CombatContext,
    FieldPreference,
    Role,
    RoleProfile,
)


class Nanally(BaseChar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def describe_role(self):
        return RoleProfile(
            role=Role.MAIN_DPS,
            field_preference=FieldPreference.MAIN_DPS,
            max_field_time=1.5,
        )

    def combat_plan(self, context):
        skill = self.click_skill_action(reason="Nanally skill available")
        ultimate = self.click_ultimate_action(reason="Nanally ultimate available")

        def entry():
            skill_result = yield skill
            if skill_result and self.ultimate_available():
                self.sleep(0.6)

            ultimate_result = yield ultimate
            if ultimate_result:
                self.perform_in_ult(context)

        return self.plan(
            skill,
            ultimate,
            entry=entry,
        )

    def perform_in_ult(self, context: CombatContext = None):
        start = time.time()
        skill_used = False
        while (elapsed := time.time() - start) < 6:
            if elapsed > 1 and not self.ultimate_available(False):
                break
            if not skill_used:
                skill_used = self._try_skill_during_ultimate(context)
            self.normal_attack()
            self.sleep(0.2)
        return skill_used

    def _try_skill_during_ultimate(self, context: CombatContext = None):
        if context is not None and not context.can_execute_action(
            self,
            slot=ActionSlot.SKILL,
        ):
            self.logger.debug("not allow skill")
            return False

        clicked = self.click_skill()
        return clicked
    
    def on_combat_end(self, chars):
        self.switch_other_char()
