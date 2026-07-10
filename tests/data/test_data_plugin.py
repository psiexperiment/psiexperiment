import pytest

from psi.data.plugin import DataPlugin


class _NoSuchSourceSink:

    def get_source(self, name):
        raise AttributeError(name)


class _UnsupportedSink:

    def get_source(self, name):
        raise NotImplementedError


class _BrokenSink:

    def get_source(self, name):
        raise RuntimeError('boom')


def test_find_source_skips_sinks_without_source():
    plugin = DataPlugin()
    plugin._sinks = {'a': _NoSuchSourceSink(), 'b': _UnsupportedSink()}
    with pytest.raises(AttributeError, match='Could not find source'):
        plugin.find_source('missing')


def test_find_source_propagates_real_errors():
    # Regression: find_source used a bare except that swallowed *any*
    # exception from a sink and reported a misleading "not found".
    plugin = DataPlugin()
    plugin._sinks = {'bad': _BrokenSink()}
    with pytest.raises(RuntimeError, match='boom'):
        plugin.find_source('anything')
