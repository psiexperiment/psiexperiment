'''
Tests for :func:`psi.application.load_io_manifest`.

A caller can reference an IO manifest either as a ``.enaml`` file (optionally
with an explicit ``::ClassName`` suffix, defaulting to ``IOManifest``) or as
a dotted Python module path (``pkg.mod.ClassName``). The '.enaml' check must
run on the path *after* stripping any '::ClassName' suffix -- checking the
un-split string would never match once a class name is appended, since
'foo.enaml::IOManifest' does not itself end in '.enaml' (see cftscal's
custom IO manifest picker, which always appends '::ClassName').
'''
from psi import application


def test_bare_enaml_path_defaults_to_iomanifest(monkeypatch):
    calls = []
    monkeypatch.setattr(
        application, 'load_manifest_from_file',
        lambda path, name: calls.append((path, name)) or 'klass',
    )
    result = application.load_io_manifest('C:/rig/io.enaml')
    assert calls == [('C:/rig/io.enaml', 'IOManifest')]
    assert result == 'klass'


def test_enaml_path_with_explicit_class(monkeypatch):
    calls = []
    monkeypatch.setattr(
        application, 'load_manifest_from_file',
        lambda path, name: calls.append((path, name)) or 'klass',
    )
    result = application.load_io_manifest('C:/rig/io.enaml::CustomManifest')
    assert calls == [('C:/rig/io.enaml', 'CustomManifest')]
    assert result == 'klass'


def test_dotted_module_path(monkeypatch):
    calls = []
    monkeypatch.setattr(
        application, 'load_manifest',
        lambda path: calls.append(path) or 'klass',
    )
    result = application.load_io_manifest('psilbhb.io.badger.IOManifest')
    assert calls == ['psilbhb.io.badger.IOManifest']
    assert result == 'klass'


def test_none_uses_default_io(monkeypatch):
    monkeypatch.setattr(application, 'get_default_io', lambda: 'C:/default/io.enaml')
    calls = []
    monkeypatch.setattr(
        application, 'load_manifest_from_file',
        lambda path, name: calls.append((path, name)) or 'klass',
    )
    result = application.load_io_manifest(None)
    assert calls == [('C:/default/io.enaml', 'IOManifest')]
    assert result == 'klass'


def test_path_object_is_coerced_to_str(monkeypatch):
    from pathlib import Path
    calls = []
    monkeypatch.setattr(
        application, 'load_manifest_from_file',
        lambda path, name: calls.append((path, name)) or 'klass',
    )
    application.load_io_manifest(Path('C:/rig/io.enaml'))
    assert calls[0][0] == str(Path('C:/rig/io.enaml'))
    assert calls[0][1] == 'IOManifest'
