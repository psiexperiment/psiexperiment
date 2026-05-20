"""Tests for psi.experiment.preferences."""
import pytest
from atom.api import Atom, Int, Str

from psi.experiment.preferences import (
    ItemPreferences, PluginPreferences, Preferences,
)


class _Holder(Atom):
    a = Int(1)
    b = Str('hello')
    c = Int(0)


# -------- ItemPreferences --------

def test_item_preferences_get_returns_auto_save_subset():
    obj = _Holder(a=42, b='world')
    prefs = ItemPreferences(item=obj, auto_save=['a', 'b'])
    assert prefs.get_preferences(workbench=None) == {'a': 42, 'b': 'world'}


def test_item_preferences_get_skips_unsaved_members():
    obj = _Holder(a=1, c=99)
    prefs = ItemPreferences(item=obj, auto_save=['a'])
    # 'c' is not in auto_save — must be excluded.
    result = prefs.get_preferences(workbench=None)
    assert 'c' not in result
    assert result == {'a': 1}


def test_item_preferences_set_updates_object():
    obj = _Holder()
    prefs = ItemPreferences(item=obj, auto_save=['a', 'b'])
    prefs.set_preferences(workbench=None, preferences={'a': 7, 'b': 'set'})
    assert obj.a == 7
    assert obj.b == 'set'


def test_item_preferences_set_ignores_unknown_attrs(caplog):
    obj = _Holder()
    prefs = ItemPreferences(item=obj, auto_save=['a'])
    # An unknown attribute name is logged at warn-level but does not raise.
    prefs.set_preferences(
        workbench=None,
        preferences={'a': 5, 'nonexistent_attr': 1},
    )
    assert obj.a == 5


# -------- PluginPreferences --------

def test_plugin_preferences_uses_workbench_lookup():
    obj = _Holder(a=11)

    class FakeWorkbench:
        def get_plugin(self, plugin_id):
            assert plugin_id == 'psi.fake'
            return obj

    prefs = PluginPreferences(plugin_id='psi.fake', auto_save=['a'])
    assert prefs.get_preferences(FakeWorkbench()) == {'a': 11}


def test_plugin_preferences_set_through_workbench():
    obj = _Holder()

    class FakeWorkbench:
        def get_plugin(self, plugin_id):
            return obj

    prefs = PluginPreferences(plugin_id='psi.fake', auto_save=['a', 'b'])
    prefs.set_preferences(FakeWorkbench(), {'a': 21, 'b': 'plugin'})
    assert obj.a == 21
    assert obj.b == 'plugin'


# -------- Preferences base class abstract methods --------

def test_preferences_base_get_object_raises():
    prefs = Preferences(name='base')
    with pytest.raises(NotImplementedError):
        prefs.get_object(workbench=None)


def test_preferences_base_get_set_preferences_raise():
    prefs = Preferences(name='base')
    with pytest.raises(NotImplementedError):
        prefs.get_preferences(workbench=None)
    with pytest.raises(NotImplementedError):
        prefs.set_preferences(workbench=None, preferences={})
