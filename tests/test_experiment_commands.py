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
    `get_default_path` builds its setting name at run time, as
    `PSI_<WHICH>_ROOT`. That spelling appears nowhere as a literal, so
    renaming either setting is invisible to a search for it and would
    only surface on the next experiment launch, where the default layout
    and preferences are loaded (see `PSIWorkbench.start_workspace`).
    Resolving both against the real configuration is what makes that
    drift fail here instead.
    '''

    def _configure(self, tmp_path, monkeypatch):
        from psi import config as psi_config
        from psi import runtime as psi_runtime

        monkeypatch.setenv('PSI_CONFIG_FILE', str(tmp_path / 'config.toml'))
        monkeypatch.setenv('PSI_BASE_DIRECTORY', str(tmp_path / 'base'))
        psi_config.reload_config()
        psi_runtime.set_runtime('EXPERIMENT', 'demo_experiment')

    @pytest.mark.parametrize('which', ['layout', 'preferences'])
    def test_resolves_against_real_settings(self, which, tmp_path,
                                            monkeypatch):
        from psi.experiment.experiment_commands import get_default_path

        self._configure(tmp_path, monkeypatch)
        try:
            path = Path(get_default_path(which))
        finally:
            from psi import runtime as psi_runtime
            psi_runtime.clear_runtime()

        assert path.parent.name == which
        assert path.name == 'demo_experiment'
        assert path.is_dir()

    @pytest.mark.parametrize('which', ['layout', 'preferences'])
    def test_filename_sits_under_that_path(self, which, tmp_path,
                                           monkeypatch):
        from psi.experiment.experiment_commands import (
            get_default_filename, get_default_path
        )

        self._configure(tmp_path, monkeypatch)
        try:
            filename = Path(get_default_filename(which))
            expected = Path(get_default_path(which)) / f'default.{which}'
        finally:
            from psi import runtime as psi_runtime
            psi_runtime.clear_runtime()

        assert filename == expected
