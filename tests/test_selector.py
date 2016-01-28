import unittest

import numpy as np

from psiexperiment.parameter import Parameter
from psiexperiment.selector import SequenceSelector


class TestSettingSequence(unittest.TestCase):

    def setUp(self):
        self.parameters = [
            Parameter('a', np.float32, expression='1'),
            Parameter('b', np.float32, expression='1'),
            Parameter('c', np.float32, expression='1'),
            Parameter('d', np.float32, expression='1'),
            Parameter('e', np.float32, expression='1'),
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
