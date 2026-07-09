
from src.char.BaseChar import BaseChar
from src.combat.planner import FieldPreference, Role, RoleProfile


class Fadia(BaseChar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def describe_role(self):
        return RoleProfile(
            role=Role.SUB_DPS,
            field_preference=FieldPreference.SUB_DPS,
        )

    def combat_plan(self, context):
        ultimate = self.click_ultimate_action()
        skill = self.click_skill_action()

        def entry():
            ultimate_result = yield ultimate
            if not ultimate_result:
                yield skill

        return self.plan(ultimate, skill, entry=entry)
