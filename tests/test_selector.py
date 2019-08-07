import pytest

import unittest
from copy import deepcopy

import numpy as np

from psi.context.api import Parameter
from psi.context.selector import SequenceSelector


class TestSettingSequence(unittest.TestCase):

    def setUp(self):
        self.parameters = [
            Parameter(name='a', default=1.0),
            Parameter(name='b', default=1.0),
            Parameter(name='c', default=1.0),
            Parameter(name='d', default=1.0),
            Parameter(name='e', default=1.0),
        ]
        self.selector = SequenceSelector()
        for parameter in self.parameters:
            self.selector.append_item(parameter.name)
        self.names = [p.name for p in self.parameters]

    def test_selector_append(self):
        self.assertEqual(len(self.selector.context_items), len(self.parameters))

    def test_selector_remove(self):
        which = 1
        self.selector.remove_item(self.parameters[which].name)
        self.assertEqual(len(self.selector.context_items), len(self.parameters)-1)
        expected_names = self.names[:]
        expected_names.pop(which)
        self.assertEqual(expected_names, self.selector.context_items)

    @pytest.mark.skip(reason='requires context plugin to be present')
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
