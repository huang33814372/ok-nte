from src.char.BaseChar import BaseChar
from src.combat.planner import ActionSlot, CombatContext, FieldPreference, Role, RoleProfile


class Zero(BaseChar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def describe_role(self):
        return RoleProfile(
            role=Role.SUB_DPS,
            field_preference=FieldPreference.SUB_DPS,
            max_field_time=1.0,
        )

    def combat_plan(self, context):
        return self.plan(
            self.click_ultimate_action(),
            self.click_skill_action(
                can_execute=self.should_use_skill,
            ),
        )

    def should_use_skill(self, context: CombatContext = None):
        return (
            not self.has_element_reaction_teammate()
            or not self.is_cycle_full()
            or (
                context is not None
                and context.strict_route_wants_action(self, slot=ActionSlot.SKILL)
            )
        )

    def click_skill(self, *args, **kwargs):
        ret = super().click_skill(*args, **kwargs)
        if ret:
            if not self.task.wait_until(
                self.is_cycle_full,
                time_out=1.25,
                raise_if_not_found=False,
            ):
                self.logger.info("cycle not full after Zero skill")
        return ret
