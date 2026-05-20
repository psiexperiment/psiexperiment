"""Tests for psi.experiment.plugin helpers and ExperimentPlugin internals.

These run against ExperimentPlugin directly without a workbench. We bypass
the lifecycle by setting private dicts manually.
"""
import pytest

from psi.experiment.metadata_item import MetadataItem
from psi.experiment.plugin import (
    ExperimentPlugin, fix_legacy_toolbar_layout,
)


# -------- fix_legacy_toolbar_layout --------

def test_fix_legacy_tuple_layout():
    out = fix_legacy_toolbar_layout((100, 200, True))
    assert out == {'x': 100, 'y': 200, 'floating': True}


def test_fix_legacy_dict_layout_passthrough():
    layout = {'x': 0, 'y': 0, 'floating': False, 'orientation': 'horizontal'}
    assert fix_legacy_toolbar_layout(layout) is layout


def test_fix_legacy_empty_dict_passthrough():
    assert fix_legacy_toolbar_layout({}) == {}


# -------- metadata_to_dict --------

def test_metadata_to_dict_simple_keys():
    plugin = ExperimentPlugin()
    plugin._metadata_items = {
        'subject': MetadataItem(name='subject', value='S001'),
        'session': MetadataItem(name='session', value='001'),
    }
    result = plugin.metadata_to_dict()
    assert result == {'subject': 'S001', 'session': '001'}


def test_metadata_to_dict_dotted_names_become_nested():
    plugin = ExperimentPlugin()
    plugin._metadata_items = {
        'animal.id': MetadataItem(name='animal.id', value='A1'),
        'animal.species': MetadataItem(name='animal.species', value='mouse'),
        'experimenter': MetadataItem(name='experimenter', value='bb'),
    }
    result = plugin.metadata_to_dict()
    assert result == {
        'animal': {'id': 'A1', 'species': 'mouse'},
        'experimenter': 'bb',
    }


def test_metadata_to_dict_deep_nesting():
    plugin = ExperimentPlugin()
    plugin._metadata_items = {
        'a.b.c': MetadataItem(name='a.b.c', value='deep'),
    }
    result = plugin.metadata_to_dict()
    assert result == {'a': {'b': {'c': 'deep'}}}


def test_metadata_to_dict_empty():
    plugin = ExperimentPlugin()
    plugin._metadata_items = {}
    assert plugin.metadata_to_dict() == {}


# -------- remap helpers default to identity --------

def test_remap_layout_default_identity():
    plugin = ExperimentPlugin()
    assert plugin.remap_layout('item.name') == 'item.name'


def test_remap_preference_default_identity():
    plugin = ExperimentPlugin()
    assert plugin.remap_preference('pref.name') == 'pref.name'
