import pytest

import unittest
from copy import deepcopy

import numpy as np

from psi.context.api import Parameter
from psi.context.selector import (
    CartesianProduct, FriendlyCartesianProduct, FriendlyCartesianProductList,
    FriendlyCartesianProductRange, SequenceSelector, SingleSetting,
)


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


# -------- SingleSetting --------

def _params():
    return (
        Parameter(name='freq', default=1000.0, dtype='float64'),
        Parameter(name='level', default=60.0, dtype='float64'),
    )


def test_single_setting_uses_param_default():
    freq, level = _params()
    s = SingleSetting()
    s.append_item(freq)
    s.append_item(level)
    it = s.get_iterator()
    setting = next(it)
    assert setting == {freq: 1000.0, level: 60.0}


def test_single_setting_set_value_updates_iterator():
    freq, _ = _params()
    s = SingleSetting()
    s.append_item(freq)
    s.set_value(freq, 2000)
    assert next(s.get_iterator())[freq] == 2000.0


# -------- CartesianProduct --------

def test_cartesian_product_orders_slowest_to_fastest():
    freq, level = _params()
    s = CartesianProduct()
    s.append_item(freq)
    s.append_item(level)
    s.add_setting(freq, 1000)
    s.add_setting(freq, 2000)
    for v in (20, 40, 60):
        s.add_setting(level, v)
    it = s.get_iterator(cycles=1)
    seq = [(d[freq], d[level]) for d in it]
    # freq is the slowest axis (added first), level the fastest.
    expected = [(1000, 20), (1000, 40), (1000, 60),
                (2000, 20), (2000, 40), (2000, 60)]
    assert seq == expected


def test_cartesian_product_finite_cycles():
    freq, = _params()[:1]
    s = CartesianProduct()
    s.append_item(freq)
    s.add_setting(freq, 1000)
    s.add_setting(freq, 2000)
    assert len(list(s.get_iterator(cycles=2))) == 4


def test_cartesian_product_empty_settings_yields_nothing():
    freq, = _params()[:1]
    s = CartesianProduct()
    s.append_item(freq)
    # No settings added — itertools.product of an empty list produces no
    # combinations, so the iterator just terminates. (warn_empty only catches
    # ValueError raised by selectors that explicitly reject empties.)
    assert list(s.get_iterator(cycles=1)) == []


# -------- SequenceSelector --------

def test_sequence_selector_ascending_sort():
    freq, = _params()[:1]
    s = SequenceSelector(order='ascending')
    s.append_item(freq)
    for v in (8000, 1000, 4000):
        s.add_setting({'freq': v})
    it = s.get_iterator(cycles=1)
    assert [d[freq] for d in it] == [1000, 4000, 8000]


def test_sequence_selector_descending_sort():
    freq, = _params()[:1]
    s = SequenceSelector(order='descending')
    s.append_item(freq)
    for v in (8000, 1000, 4000):
        s.add_setting({'freq': v})
    assert [d[freq] for d in s.get_iterator(cycles=1)] == [8000, 4000, 1000]


def test_sequence_selector_remove_setting():
    freq, = _params()[:1]
    s = SequenceSelector(order='ascending')
    s.append_item(freq)
    s.add_setting({'freq': 1000})
    s.add_setting({'freq': 2000})
    s.remove_setting(s.settings[0])
    assert len(s.settings) == 1
    assert [d[freq] for d in s.get_iterator(cycles=1)] == [2000]


def test_sequence_selector_clear_settings():
    freq, = _params()[:1]
    s = SequenceSelector(order='ascending')
    s.append_item(freq)
    s.add_setting({'freq': 1000})
    s.clear_settings()
    assert s.settings == []


def test_sequence_selector_empty_allows_empty_yields_empty_dict():
    s = SequenceSelector(order='ascending')
    # No items, no settings, allow_empty=True (the default).
    it = s.get_iterator(cycles=2, allow_empty=True)
    assert next(it) == {}


def test_sequence_selector_with_expression_uses_symbols():
    am_depth = Parameter(name='am_depth', dtype='float64')
    s = SequenceSelector(order='descending')
    s.symbols = {'dbi': lambda x: 10 ** (x / 20)}
    s.append_item(am_depth)
    for expr in ('dbi(-3)', 'dbi(0)', 'dbi(-6)'):
        s.add_setting({'am_depth': expr})
    # Sort key evaluates each expression through `symbols`.
    seq = [d[am_depth] for d in s.get_iterator(cycles=1)]
    assert seq == ['dbi(0)', 'dbi(-3)', 'dbi(-6)']


def test_sequence_selector_set_and_get_value():
    freq, = _params()[:1]
    s = SequenceSelector(order='ascending')
    s.append_item(freq)
    s.add_setting({'freq': 1000})
    s.set_value(0, freq, 2500)
    assert s.get_value(0, freq) == 2500.0


# -------- preferences --------

def test_selector_preferences_capture_context_item_order():
    freq, level = _params()
    s = SequenceSelector(order='ascending')
    s.append_item(freq)
    s.append_item(level)
    prefs = s.get_preferences()
    assert prefs['context_item_order'] == ['freq', 'level']


def test_selector_context_item_order_reorders():
    freq, level = _params()
    s = SequenceSelector(order='ascending')
    s.append_item(freq)
    s.append_item(level)
    s.context_item_order = ['level', 'freq']
    assert [i.name for i in s.context_items] == ['level', 'freq']


# -------- FriendlyCartesianProduct --------

def test_friendly_cartesian_product_range_values():
    freq = Parameter(name='freq', default=1000.0, dtype='float64')
    s = FriendlyCartesianProduct()
    s.context_detail = {'freq': {'user_friendly_name': 'frequency'}}
    s.append_item(freq)
    setting = s.get_setting(freq)
    assert isinstance(setting, FriendlyCartesianProductRange)
    setting.start = 1000
    setting.end = 3000
    setting.step = 1000
    np.testing.assert_array_equal(setting.get_values(), [1000, 2000, 3000])


def test_friendly_cartesian_product_change_to_list():
    freq = Parameter(name='freq', default=1000.0, dtype='float64')
    s = FriendlyCartesianProduct()
    s.context_detail = {'freq': {'user_friendly_name': 'frequency'}}
    s.append_item(freq)
    new_setting = s.change_setting(freq, FriendlyCartesianProductList)
    new_setting.values = [500, 1000, 2000]
    np.testing.assert_array_equal(s.get_values(freq), [500, 1000, 2000])
