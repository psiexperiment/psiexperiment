"""Tests for layout persistence in psi.experiment.experiment_commands."""
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

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


class TestDefaultPath:
    '''
    `get_default_path` resolves PSI_SETTINGS_ROOT at run time and appends
    the paradigm, so a rename of the setting is invisible to a search for
    it and would only surface on the next experiment launch, where the
    default layout and preferences are loaded (see
    `PSIWorkbench.start_workspace`). Resolving against the real
    configuration is what makes that drift fail here instead.

    The paradigm comes from the workbench rather than from a process
    global, so a stub carrying the one attribute is a faithful caller --
    `tests/workbench/test_experiment_name.py` checks that the real
    PSIWorkbench provides it.
    '''

    def _workbench(self, tmp_path, monkeypatch, name='demo_experiment'):
        monkeypatch.setenv('PSI_BASE_DIRECTORY', str(tmp_path / 'base'))
        from psi import config as psi_config
        psi_config.reload_config()
        return SimpleNamespace(experiment_name=name)

    @pytest.mark.parametrize('which', ['layout', 'preferences'])
    def test_resolves_against_real_settings(self, which, tmp_path,
                                            monkeypatch):
        from psi.experiment.experiment_commands import get_default_path

        workbench = self._workbench(tmp_path, monkeypatch)
        path = Path(get_default_path(workbench, which))

        assert path.parent.name == which
        assert path.name == 'demo_experiment'
        assert path.is_dir()

    @pytest.mark.parametrize('which', ['layout', 'preferences'])
    def test_filename_sits_under_that_path(self, which, tmp_path,
                                           monkeypatch):
        from psi.experiment.experiment_commands import (
            get_default_filename, get_default_path
        )

        workbench = self._workbench(tmp_path, monkeypatch)
        filename = Path(get_default_filename(workbench, which))
        expected = Path(get_default_path(workbench, which)) / f'default.{which}'

        assert filename == expected

    def test_no_experiment_name_is_reported(self, tmp_path, monkeypatch):
        '''
        Without a name this used to build <root>/<which>/ and create it,
        so the default preferences for every paradigm would have landed
        in one directory. It is a startup step that did not run, so say
        so rather than carrying on.
        '''
        from psi.experiment.experiment_commands import get_default_path

        workbench = self._workbench(tmp_path, monkeypatch, name='')
        with pytest.raises(ValueError, match='experiment name'):
            get_default_path(workbench, 'layout')
