"""Workbench tests for the data plugin."""
import pytest


def test_data_plugin_starts(workbench):
    plugin = workbench.get_plugin('psi.data')
    # The workbench auto-registers a baseline set of sinks (metadata,
    # calibration_data, base_store). Plot/container/viewbox collections
    # remain empty since the helper manifest doesn't contribute any.
    assert set(plugin._sinks) >= {'metadata', 'base_store', 'calibration_data'}
    assert plugin._containers == {}
    assert plugin._plots == {}


def test_find_sink_unknown_raises(workbench):
    plugin = workbench.get_plugin('psi.data')
    with pytest.raises(AttributeError, match='not available'):
        plugin.find_sink('nope')


def test_find_viewbox_unknown_raises(workbench):
    plugin = workbench.get_plugin('psi.data')
    with pytest.raises(AttributeError, match='not available'):
        plugin.find_viewbox('nope')


def test_find_plot_container_unknown_raises(workbench):
    plugin = workbench.get_plugin('psi.data')
    with pytest.raises(AttributeError, match='not available'):
        plugin.find_plot_container('nope')


def test_find_source_unknown_raises(workbench):
    plugin = workbench.get_plugin('psi.data')
    with pytest.raises(AttributeError, match='Could not find source'):
        plugin.find_source('nope')


def test_set_base_path_propagates(workbench, tmp_path):
    plugin = workbench.get_plugin('psi.data')
    # No sinks to receive; should still work and update the plugin attribute.
    plugin.set_base_path(tmp_path, is_temp=False)
    assert plugin.base_path == tmp_path
