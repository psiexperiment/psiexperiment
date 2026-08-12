"""Tests for layout persistence in psi.experiment.experiment_commands."""
import pickle
from types import SimpleNamespace

from enaml.layout.dock_layout import AreaLayout, DockLayout, ItemLayout
from enaml.layout.geometry import Rect

from psi.experiment.experiment_commands import _load_layout, _save_layout
from psi.experiment.dock_layout_serializer import dock_layout_node_to_dict


def _sample_layout():
    return {
        'geometry': Rect(0, 0, 800, 600),
        'toolbars': {
            'main': {'floating': False, 'orientation': 'horizontal',
                      'dock_area': 'top', 'x': 0, 'y': 0},
        },
        'dock_layout': DockLayout(AreaLayout(ItemLayout('microphone'))),
    }


class _FakePlugin:

    def __init__(self, layout=None):
        self._layout = layout
        self.set_layout_calls = []

    def get_layout(self):
        return self._layout

    def set_layout(self, layout):
        self.set_layout_calls.append(layout)


def _fake_event(plugin):
    workbench = SimpleNamespace(get_plugin=lambda name: plugin)
    return SimpleNamespace(workbench=workbench)


def test_save_layout_writes_text_not_binary(tmp_path):
    filename = str(tmp_path / 'test.layout')
    plugin = _FakePlugin(_sample_layout())
    _save_layout(_fake_event(plugin), filename)

    # Must be plain text (YAML), not a pickle stream.
    with open(filename, 'rb') as fh:
        assert fh.read(1) != b'\x80'
    with open(filename, 'r') as fh:
        text = fh.read()
    assert 'dock_layout' in text
    assert 'microphone' in text


def test_save_then_load_roundtrips_through_yaml(tmp_path):
    filename = str(tmp_path / 'test.layout')
    layout = _sample_layout()
    _save_layout(_fake_event(_FakePlugin(layout)), filename)

    load_plugin = _FakePlugin()
    _load_layout(_fake_event(load_plugin), filename)

    assert len(load_plugin.set_layout_calls) == 1
    loaded = load_plugin.set_layout_calls[0]
    assert loaded['geometry'] == layout['geometry']
    assert loaded['toolbars'] == layout['toolbars']
    assert dock_layout_node_to_dict(loaded['dock_layout']) == \
        dock_layout_node_to_dict(layout['dock_layout'])


def test_load_layout_falls_back_to_legacy_pickle(tmp_path):
    # Regression: old .layout files saved before the YAML switch must
    # still load.
    filename = str(tmp_path / 'legacy.layout')
    layout = _sample_layout()
    with open(filename, 'wb') as fh:
        pickle.dump(layout, fh)

    load_plugin = _FakePlugin()
    _load_layout(_fake_event(load_plugin), filename)

    loaded = load_plugin.set_layout_calls[0]
    assert loaded['geometry'] == layout['geometry']
    assert dock_layout_node_to_dict(loaded['dock_layout']) == \
        dock_layout_node_to_dict(layout['dock_layout'])
