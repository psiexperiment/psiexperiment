import unittest2 as unittest

import numpy as np

from psiexperiment.parameter import Parameter
from psiexperiment.selector import SequenceSelector
from psiexperiment.choice import ascending
from psiexperiment.controller import Controller


parameters = [
    Parameter(name='trials', expression='80', dtype=np.int32, rove=True),
    Parameter(name='level', expression='60', dtype=np.float64),
    Parameter(name='fc', expression='32e3/trials', dtype=np.float64),
]

class TestController(unittest.TestCase):

    def setUp(self):
        selector = SequenceSelector(order=ascending)
        selector.append_parameter(parameters[0])
        selector.add_setting(dict(trials=20))
        selector.add_setting(dict(trials=15))
        selector.add_setting(dict(trials=10))
        selector.add_setting(dict(trials=2))
        self.controller = Controller(parameters=parameters,
                                     selectors={'default': selector})

    def test_eval(self):
        expected = [
            dict(trials=2, level=60, fc=32e3/2),
            dict(trials=10, level=60, fc=32e3/10),
            dict(trials=15, level=60, fc=32e3/15),
            dict(trials=20, level=60, fc=32e3/20),
            dict(trials=2, level=60, fc=32e3/2),
            dict(trials=10, level=60, fc=32e3/10),
        ]
        self.controller.request_apply()
        for e in expected:
            self.controller.next_trial()
            a = self.controller.get_values()
            self.assertEqual(a, e)

        self.controller.request_apply()
        for e in expected:
            self.controller.next_trial()
            a = self.controller.get_values()
            self.assertEqual(a, e)

    def test_update(self):
        self.controller.start_experiment()
        self.assertFalse(self.controller._changes_pending)
        self.assertEqual(self.controller.get_value('level'), 60)
        self.controller.next_trial()
        self.assertFalse(self.controller._changes_pending)
        self.assertEqual(self.controller.get_value('level'), 60)

        self.controller.get_parameter('level').expression = '32'
        self.assertTrue(self.controller._changes_pending)
        self.assertEqual(self.controller.get_value('level'), 60)

        self.controller.next_trial()
        self.assertTrue(self.controller._changes_pending)
        self.assertEqual(self.controller.get_value('level'), 60)

        self.controller.request_apply()
        self.controller.next_trial()
        self.assertFalse(self.controller._changes_pending)
        self.assertEqual(self.controller.get_value('level'), 32)
