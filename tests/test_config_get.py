'''
Tests for `psi-config get`, which prints one setting's value for a script.

What matters is that stdout holds the bare value and nothing else -- a
Windows batch file captures it with ``for /f`` and cannot tell a value from
a message -- and that a failure prints nothing there and exits non-zero.
'''
import sys
from pathlib import Path

import pytest

from psi.application import _format_for_shell


class TestFormatForShell:

    def test_path_is_native(self):
        path = Path('C:/Data/psi')
        assert _format_for_shell('X', path) == str(path)

    @pytest.mark.parametrize('value, expected', [
        (True, 'true'),
        (False, 'false'),
        (5, '5'),
        (96000.0, '96000.0'),
        ('text with spaces', 'text with spaces'),
        ('', ''),
        (None, ''),
    ])
    def test_scalars(self, value, expected):
        assert _format_for_shell('X', value) == expected

    def test_list_is_comma_joined(self):
        # The same spelling `psi-config set` and the environment accept.
        assert _format_for_shell('X', ['a', 'b']) == 'a,b'
        assert _format_for_shell('X', []) == ''

    def test_table_is_refused(self):
        with pytest.raises(ValueError, match='X holds a table'):
            _format_for_shell('X', {'a': 1})

    def test_nested_list_is_refused(self):
        with pytest.raises(ValueError, match='nested'):
            _format_for_shell('X', [['a']])


class TestGetCommand:

    @pytest.fixture
    def downstream(self, tmp_path, monkeypatch):
        '''
        A package declaring settings through the psi.settings entry point,
        as cftscal and noise-exp do.
        '''
        import importlib.metadata
        from types import SimpleNamespace

        from psi import config as psi_config

        defaults = {
            'FAKEPKG_ROOT': lambda: tmp_path / 'fakepkg',
            'FAKEPKG_TABLE': lambda: {},
        }
        entry = SimpleNamespace(name='fakepkg', load=lambda: defaults)
        monkeypatch.setattr(importlib.metadata, 'entry_points',
                            lambda group=None: [entry])
        yield tmp_path / 'fakepkg'
        for name in defaults:
            psi_config._defaults.pop(name, None)

    def _get(self, monkeypatch, capsys, setting):
        from psi.application import config

        monkeypatch.setattr(sys, 'argv', ['psi-config', 'get', setting])
        config()
        return capsys.readouterr()

    def test_prints_only_the_value(self, monkeypatch, capsys):
        from psi import get_config

        out = self._get(monkeypatch, capsys, 'PSI_DATA_ROOT')
        assert out.out == f'{get_config("PSI_DATA_ROOT")}\n'

    def test_config_file_value(self, monkeypatch, capsys, tmp_path):
        from psi import save_config

        save_config({'PSI_DATA_ROOT': str(tmp_path / 'my data')})
        out = self._get(monkeypatch, capsys, 'PSI_DATA_ROOT')
        assert out.out == f'{tmp_path / "my data"}\n'

    def test_environment_wins(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv('PSI_DATA_ROOT', str(tmp_path / 'from env'))
        out = self._get(monkeypatch, capsys, 'PSI_DATA_ROOT')
        assert out.out == f'{tmp_path / "from env"}\n'

    def test_downstream_setting(self, downstream, monkeypatch, capsys):
        # Known only through the entry point, since psi-config imports psi
        # alone.
        out = self._get(monkeypatch, capsys, 'FAKEPKG_ROOT')
        assert out.out == f'{downstream}\n'

    def test_unknown_setting_fails_quietly(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as info:
            self._get(monkeypatch, capsys, 'NOT_A_SETTING')
        assert 'NOT_A_SETTING is not a known setting' in str(info.value.code)
        assert capsys.readouterr().out == ''

    def test_table_fails_quietly(self, downstream, monkeypatch, capsys):
        with pytest.raises(SystemExit) as info:
            self._get(monkeypatch, capsys, 'FAKEPKG_TABLE')
        assert 'holds a table' in str(info.value.code)
        assert capsys.readouterr().out == ''
