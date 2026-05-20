from types import SimpleNamespace

import pytest

from psi.experiment.paradigm_description import (
    ParadigmManager, ParadigmNotFound, PluginDescription,
)


@pytest.fixture
def manager():
    return ParadigmManager()


def _make_paradigm(name, experiment_type='ear'):
    return SimpleNamespace(name=name, experiment_type=experiment_type)


def test_register_stores_paradigm(manager):
    p = _make_paradigm('foo')
    manager.register(p)
    assert manager.paradigms == {'foo': p}


def test_list_paradigms_no_filter(manager):
    manager.register(_make_paradigm('a'))
    manager.register(_make_paradigm('b'))
    names = [p.name for p in manager.list_paradigms()]
    assert sorted(names) == ['a', 'b']


def test_list_paradigms_filtered_by_experiment_type(manager):
    manager.register(_make_paradigm('a', 'ear'))
    manager.register(_make_paradigm('b', 'animal'))
    manager.register(_make_paradigm('c', 'ear'))
    names = sorted(p.name for p in manager.list_paradigms(experiment_type='ear'))
    assert names == ['a', 'c']


def test_list_paradigm_names_is_sorted(manager):
    for name in ['zebra', 'apple', 'mango']:
        manager.register(_make_paradigm(name))
    assert manager.list_paradigm_names() == ['apple', 'mango', 'zebra']


def test_get_paradigm_simple_name(manager):
    p = _make_paradigm('foo')
    manager.register(p)
    assert manager.get_paradigm('foo') is p


def test_get_paradigm_missing_raises(manager):
    with pytest.raises(ParadigmNotFound):
        manager.get_paradigm('missing')


def test_get_paradigm_dotted_name_strips_module(manager):
    # Dotted names ask the manager to import the module first, then look up
    # the paradigm by its short name. We register the short name directly.
    p = _make_paradigm('foo')
    manager.register(p)
    # The module portion must resolve; use a real stdlib module that exists.
    assert manager.get_paradigm('json.foo') is p


# -------- PluginDescription --------

class _FakeManifest:
    required = True
    id = 'auto.id'
    title = 'Auto Title'


def test_plugin_description_uses_manifest_defaults_when_unset(monkeypatch):
    monkeypatch.setattr(
        'psi.experiment.paradigm_description.load_manifest',
        lambda path: lambda **kw: _FakeManifest(),
    )
    p = PluginDescription(manifest='psi.fake.Manifest')
    assert p.required is True
    assert p.id == 'auto.id'
    assert p.title == 'Auto Title'


def test_plugin_description_explicit_values_win(monkeypatch):
    monkeypatch.setattr(
        'psi.experiment.paradigm_description.load_manifest',
        lambda path: lambda **kw: _FakeManifest(),
    )
    p = PluginDescription(
        manifest='psi.fake.Manifest',
        required=False,
        id='custom.id',
        title='Custom',
    )
    assert p.required is False
    assert p.id == 'custom.id'
    assert p.title == 'Custom'


def test_plugin_description_lazy_manifest_load(monkeypatch):
    call_count = {'n': 0}

    def fake_load(path):
        call_count['n'] += 1
        return lambda **kw: _FakeManifest()

    monkeypatch.setattr(
        'psi.experiment.paradigm_description.load_manifest', fake_load
    )
    # All three defaults provided -- manifest is never accessed.
    p = PluginDescription(
        manifest='psi.fake.Manifest',
        required=False, id='x', title='y',
    )
    assert call_count['n'] == 0
    # Accessing the manifest property triggers the load.
    _ = p.manifest
    assert call_count['n'] == 1
    # Subsequent access is cached.
    _ = p.manifest
    assert call_count['n'] == 1
