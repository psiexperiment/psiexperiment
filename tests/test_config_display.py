'''
Tests for how `psi-config show` renders settings.

This listing is the only diagnostic a user has for "why is this setting
not what I put in the file?", and on a frozen application it is the only
one available at all. It stops being usable the moment one nested table
-- cftscal keeps its whole per-plugin GUI state in one -- runs off the
side of the screen and buries everything else.
'''
from pathlib import Path

import pytest

from psi.config_cli import _group_label, _render_setting


class TestRenderSetting:

    def test_path_is_pasteable(self):
        # Not WindowsPath('C:/data'): this is read to check a location. As
        # the platform writes it (C:\data on Windows).
        path = Path('C:/data')
        assert _render_setting(path) == str(path)
        assert 'Path(' not in _render_setting(path)

    def test_string_is_unquoted(self):
        assert _render_setting('info') == 'info'

    def test_empty_string_is_visible(self):
        # '' would render as nothing at all, which reads as a bug.
        assert _render_setting('') == '(empty)'

    @pytest.mark.parametrize('value,expected', [
        (6, '6'), (96000.0, '96000.0'), (True, 'True'), (False, 'False'),
    ])
    def test_scalars(self, value, expected):
        assert _render_setting(value) == expected

    def test_empty_container(self):
        assert _render_setting([]) == '(none)'
        assert _render_setting({}) == '(none)'

    def test_list_of_scalars_is_joined(self):
        assert _render_setting(['cfts.paradigms', 'noise_exp.paradigms']) \
            == 'cfts.paradigms, noise_exp.paradigms'

    def test_table_is_summarized_by_its_keys(self):
        '''
        The case that prompted this: CFTSCAL_PLUGIN holds a table per
        plugin, and printing it inline pushed every other setting's value
        off the screen.
        '''
        value = {'microphone': {'selected_input': 'ai0', 'gain': 20},
                 'starship': {'selected_coupler': 'tube-2mm'}}
        assert _render_setting(value) == '(2 entries: microphone, starship)'

    def test_single_entry_is_singular(self):
        assert _render_setting({'microphone': {}}) == '(1 entry: microphone)'

    def test_nested_list_is_counted_not_joined(self):
        assert _render_setting([{'a': 1}, {'b': 2}]) == '(2 items)'

    def test_verbose_prints_containers_in_full(self):
        value = {'microphone': {'gain': 20}}
        assert _render_setting(value, verbose=True) == repr(value)

    def test_summary_fits_on_a_line(self):
        '''
        A table with many plugins must still summarize to something that
        does not wrap -- the whole point of summarizing.
        '''
        value = {f'plugin_{i}': {'a': 1} for i in range(3)}
        assert len(_render_setting(value)) < 60


class TestGroupLabel:

    def test_common_prefix_of_several(self):
        assert _group_label(['PSI_DATA_ROOT', 'PSI_LOG_ROOT']) == 'PSI'

    def test_two_segment_prefix(self):
        # Splitting on the first underscore would label this 'NOISE'.
        assert _group_label(['NOISE_EXP_MAX_ANIMALS',
                             'NOISE_EXP_PREFERENCE']) == 'NOISE_EXP'

    def test_never_consumes_the_last_segment(self):
        # A lone setting is grouped by its package, not by its own name.
        assert _group_label(['CFTS_ROOT']) == 'CFTS'

    def test_stops_where_the_names_diverge(self):
        assert _group_label(['PSI_BASE_DIRECTORY', 'PSI_HOSTNAME']) == 'PSI'

    def test_longer_prefix_wins_when_all_agree(self):
        assert _group_label(['CFTSCAL_DEVICE_NAME',
                             'CFTSCAL_DEVICE_HOSTAPI']) == 'CFTSCAL_DEVICE'


class TestRegisterDownstreamSettings:
    '''
    A package registers its settings when imported, and psi-config
    imports only psi -- so without discovery, every CFTSCAL_ key in the
    configuration file is reported as belonging to no setting at all.

    Discovery is by entry point rather than from the configuration file:
    cftscal and cfts name their paradigms by fully-qualified path and
    register them when their own GUI starts, so neither ever appears in
    PSI_PARADIGM_DESCRIPTIONS.
    '''

    def _entry(self, name, value):
        from types import SimpleNamespace
        return SimpleNamespace(name=name, load=lambda: value)

    def _call(self, monkeypatch, entries):
        '''
        Run the helper with `entry_points` replaced.

        It imports entry_points inside the function, so the patch has to
        land on importlib.metadata itself.
        '''
        import importlib.metadata

        from psi import config_cli

        monkeypatch.setattr(importlib.metadata, 'entry_points',
                            lambda group=None: entries)
        return config_cli._register_downstream_settings()

    def test_registers_what_the_entry_point_returns(self, monkeypatch):
        from psi import config as psi_config

        table = {'FAKEPKG_SETTING': psi_config.Setting(str, 'value')}
        try:
            loaded = self._call(monkeypatch, [self._entry('fakepkg', table)])
            assert loaded == ['fakepkg']
            assert psi_config.get_config('FAKEPKG_SETTING') == 'value'
        finally:
            psi_config._defaults.pop('FAKEPKG_SETTING', None)

    def test_one_bad_package_does_not_stop_the_rest(self, monkeypatch):
        '''
        psi-config show is what somebody runs when something is already
        broken, so a package that will not import must not take the whole
        listing down with it.
        '''
        from types import SimpleNamespace

        from psi import config as psi_config

        def boom():
            raise ImportError('no such module')

        good = {'FAKEPKG_SETTING': psi_config.Setting(str, 'value')}
        entries = [
            SimpleNamespace(name='broken', load=boom),
            self._entry('fakepkg', good),
        ]
        try:
            loaded = self._call(monkeypatch, entries)
            assert loaded == ['fakepkg']
            assert psi_config.get_config('FAKEPKG_SETTING') == 'value'
        finally:
            psi_config._defaults.pop('FAKEPKG_SETTING', None)

    def test_no_entry_points_is_not_an_error(self, monkeypatch):
        assert self._call(monkeypatch, []) == []


class TestCreateCommandsMakeDownstreamFolders:
    '''
    `psi-config create` and `create-folders` build the directory tree by
    walking the registered settings, so they have to load the downstream
    packages the same way `show` does. Without that they make psi's own
    directories and silently skip CFTSCAL_ROOT and every other root a
    package declares -- on the two commands whose whole job is laying out
    the tree for a new rig.
    '''

    @pytest.fixture
    def downstream_root(self, tmp_path, monkeypatch):
        import importlib.metadata
        from types import SimpleNamespace

        from psi import config as psi_config

        root = tmp_path / 'fakepkg'
        table = {'FAKEPKG_ROOT': psi_config.Setting(Path, root)}
        entry = SimpleNamespace(name='fakepkg', load=lambda: table)
        monkeypatch.setattr(importlib.metadata, 'entry_points',
                            lambda group=None: [entry])
        yield root
        psi_config._defaults.pop('FAKEPKG_ROOT', None)

    def _run(self, monkeypatch, *argv):
        import sys

        from psi.config_cli import main as config

        monkeypatch.setattr(sys, 'argv', ['psi-config', *argv])
        config()

    def test_create(self, downstream_root, tmp_path, monkeypatch):
        self._run(monkeypatch, 'create',
                  '--base-directory', str(tmp_path / 'base'))
        assert (tmp_path / 'base' / 'data').is_dir()
        assert downstream_root.is_dir()

    def test_create_folders(self, downstream_root, monkeypatch):
        self._run(monkeypatch, 'create-folders')
        assert downstream_root.is_dir()


class TestDescribeConfigFile:
    '''
    `psi-config show` must say which file it read and whether
    PSI_CONFIG_FILE chose it: a variable left set in a batch file or a
    terminal silently points psi at a file other than the one being edited.
    '''

    def test_default_location_explains_override(self, monkeypatch):
        from psi.config_cli import describe_config_file
        monkeypatch.delenv('PSI_CONFIG_FILE', raising=False)
        lines = describe_config_file()
        assert str(Path('~/psi/config.toml').expanduser()) in lines[0]
        assert 'default location' in lines[1]
        assert 'PSI_CONFIG_FILE' in lines[1]

    def test_reports_override(self, monkeypatch, tmp_path):
        from psi.config_cli import describe_config_file
        path = tmp_path / 'rig.toml'
        path.write_text('', encoding='utf-8')
        monkeypatch.setenv('PSI_CONFIG_FILE', str(path))
        lines = describe_config_file()
        assert lines[0] == f'Configuration file: {path} (exists)'
        assert 'Chosen by the PSI_CONFIG_FILE' in lines[1]

    def test_empty_variable_is_not_an_override(self, monkeypatch):
        # get_config_file treats an empty PSI_CONFIG_FILE as unset; the
        # description must agree rather than claim an override.
        from psi.config_cli import describe_config_file
        monkeypatch.setenv('PSI_CONFIG_FILE', '')
        assert 'default location' in describe_config_file()[1]

    def test_help_explains_override(self, capsys):
        from psi.config_cli import build_parser
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--help'])
        out = capsys.readouterr().out
        assert 'PSI_CONFIG_FILE' in out
        # The example commands keep their own lines.
        assert '\n  PowerShell:' in out
        assert 'conda env config vars set PSI_CONFIG_FILE=' in out
        assert 'Scripts\\activate.bat' in out
        assert 'setx PSI_CONFIG_FILE' in out
