# Test case
import unittest

from ok.test.TaskTestCase import TaskTestCase

from src.char.BaseChar import Element
from src.config import config
from src.tasks.trigger.AutoCombatTask import AutoCombatTask

TEAM = [0, 1, 2, 3]

class TestAutoCombatTask(TaskTestCase):
    task_class = AutoCombatTask

    config = config

    def test(self):
        # Create a BattleReport object
        self.set_image('tests/images/elements_1.png')
        expected = [Element.WHITE, Element.GREEN, Element.GREEN, Element.WHITE]
        chars_elements = self.task.load_chars_element(TEAM)
        for i in TEAM:
            result = chars_elements.get(i, Element.DEFAULT)
            self.assertEqual(expected[i], result)



if __name__ == '__main__':
    unittest.main()
