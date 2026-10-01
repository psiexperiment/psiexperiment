'''
Tests for `psi-config set`.
'''
import sys
import tomllib

import pytest

from psi import config as psi_config


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / 'config.toml'
    monkeypatch.setenv('PSI_CONFIG_FILE', str(path))
    psi_config.reload_config()
    yield path
    psi_config.reload_config()


@pytest.fixture
def declared():
    names = {
        'PSI_T_RATE': psi_config.Setting(float, 48000),
        'PSI_T_FLAG': psi_config.Setting(bool, False),
        'PSI_T_LIST': psi_config.Setting(list, []),
        'PSI_T_TABLE': psi_config.Setting(dict, {}),
    }
    psi_config.register_defaults(names)
    yield names
    for name in names:
        psi_config._defaults.pop(name, None)


def _set(monkeypatch, *argv):
    from psi.config_cli import main as config

    monkeypatch.setattr(sys, 'argv', ['psi-config', 'set', *argv])
    config()


def _written(path):
    return tomllib.loads(path.read_text(encoding='utf-8'))


def test_misspelled_name_is_refused(config_file, monkeypatch):
    with pytest.raises(SystemExit) as info:
        _set(monkeypatch, 'PSI_DATA_ROT', 'D:/x')
    assert 'Did you mean PSI_DATA_ROOT?' in str(info.value.code)
    assert not config_file.exists()


def test_force_writes_an_unknown_name(config_file, monkeypatch):
    _set(monkeypatch, 'PSI_NOT_INSTALLED_HERE', 'x', '--force')
    assert _written(config_file) == {'PSI_NOT_INSTALLED_HERE': 'x'}


def test_values_are_written_as_their_type(config_file, declared,
                                          monkeypatch):
    _set(monkeypatch, 'PSI_T_RATE', '96000')
    _set(monkeypatch, 'PSI_T_FLAG', 'yes')
    _set(monkeypatch, 'PSI_T_LIST', 'a, b')
    _set(monkeypatch, 'PSI_DATA_ROOT', 'D:/data')
    assert _written(config_file) == {
        'PSI_T_RATE': 96000.0,
        'PSI_T_FLAG': True,
        'PSI_T_LIST': ['a', 'b'],
        'PSI_DATA_ROOT': 'D:/data',
    }


def test_invalid_value_is_not_written(config_file, declared, monkeypatch):
    with pytest.raises(SystemExit) as info:
        _set(monkeypatch, 'PSI_T_FLAG', 'maybe')
    assert 'PSI_T_FLAG' in str(info.value.code)
    assert not config_file.exists()


def test_table_is_refused(config_file, declared, monkeypatch):
    with pytest.raises(SystemExit, match='holds a table'):
        _set(monkeypatch, 'PSI_T_TABLE', 'x')
