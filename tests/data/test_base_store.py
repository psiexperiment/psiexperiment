"""Tests for the BaseStore class (psi.data.sinks.base_store)."""
import enaml
import pytest


with enaml.imports():
    from psi.data.sinks.base_store import BaseStore


class _RecordingSubstore:
    def __init__(self):
        self.flushed = False

    def flush(self):
        self.flushed = True


def test_get_filename_joins_base_path_and_suffix(tmp_path):
    store = BaseStore()
    store.set_base_path(tmp_path, is_temp=False)
    out = store.get_filename('signal', suffix='.bin')
    assert out == (tmp_path / 'signal').with_suffix('.bin')


def test_set_base_path_records_is_temp(tmp_path):
    store = BaseStore()
    store.set_base_path(tmp_path, is_temp=True)
    assert store.is_temp is True
    assert store.base_path == tmp_path


def test_flush_propagates_to_substores(tmp_path):
    store = BaseStore()
    store.set_base_path(tmp_path, is_temp=False)
    a, b = _RecordingSubstore(), _RecordingSubstore()
    store._stores['a'] = a
    store._stores['b'] = b
    store.flush()
    assert a.flushed and b.flushed


def test_create_ai_methods_raise_not_implemented():
    store = BaseStore()
    with pytest.raises(NotImplementedError):
        store.create_ai_continuous('sig', 100, 'd', {})
    with pytest.raises(NotImplementedError):
        store.create_ai_epochs('sig', 100, 'd', {})
    with pytest.raises(NotImplementedError):
        store.process_ai_continuous('sig', None)
    with pytest.raises(NotImplementedError):
        store.process_ai_epochs('sig', None)
    with pytest.raises(NotImplementedError):
        store.get_source('sig')
