import unittest
from copy import deepcopy

import numpy as np

from psi.context.api import RovingParameter
from psi.context.selector import SequenceSelector


class TestSettingSequence(unittest.TestCase):

    def setUp(self):
        self.parameters = [
            RovingParameter(name='a', dtype=np.float32, expression='1'),
            RovingParameter(name='b', dtype=np.float32, expression='1'),
            RovingParameter(name='c', dtype=np.float32, expression='1'),
            RovingParameter(name='d', dtype=np.float32, expression='1'),
            RovingParameter(name='e', dtype=np.float32, expression='1'),
        ]
        self.selector = SequenceSelector()
        for parameter in self.parameters:
            self.selector.append_parameter(parameter)
        self.names = [p.name for p in self.parameters]

    def test_selector_append(self):
        self.assertEqual(len(self.selector.parameters), len(self.parameters))

    def test_selector_remove(self):
        which = 1
        self.selector.remove_parameter(self.parameters[which])
        self.assertEqual(len(self.selector.parameters), len(self.parameters)-1)
        expected_names = self.names[:]
        expected_names.pop(which)
        names = [p.name for p in self.selector.parameters]
        self.assertEqual(expected_names, names)

    def test_selector_move(self):
        self.selector.move_parameter(self.parameters[1])
        names = [p.name for p in self.selector.parameters]
        expected_names = [u'b', u'a', u'c', u'd', u'e']
        self.assertEqual(expected_names, names)

        self.selector.move_parameter(self.parameters[0], self.parameters[-1])
        names = [p.name for p in self.selector.parameters]
        expected_names = [u'b', u'c', u'd', u'e', u'a']
        self.assertEqual(expected_names, names)

        self.selector.move_parameter(self.parameters[0], self.parameters[1])
        names = [p.name for p in self.selector.parameters]
        expected_names = [u'b', u'a', u'c', u'd', u'e']
        self.assertEqual(expected_names, names)

    def test_deepcopy_equality(self):
        c1 = deepcopy(self.selector.__getstate__())
        self.selector.add_setting({'a': 4, 'b': 5})
        self.selector.add_setting({'a': 7})
        c2 = deepcopy(self.selector.__getstate__())
        self.selector.set_value(0, 'a', 1)
        c3 = deepcopy(self.selector.__getstate__())
        c4 = deepcopy(self.selector.__getstate__())
        self.assertFalse(c1 == c2)
        self.assertFalse(c2 == c3)
        self.assertTrue(c3 == c4)
