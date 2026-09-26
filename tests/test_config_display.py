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

from psi.application import _group_label, _render_setting


class TestRenderSetting:

    def test_path_is_pasteable(self):
        # Not WindowsPath('C:/data'): this is read to check a location.
        assert _render_setting(Path('C:/data')) == 'C:\\data'

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

        from psi import application

        monkeypatch.setattr(importlib.metadata, 'entry_points',
                            lambda group=None: entries)
        return application._register_downstream_settings()

    def test_registers_what_the_entry_point_returns(self, monkeypatch):
        from psi import config as psi_config

        table = {'FAKEPKG_SETTING': lambda: 'value'}
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

        good = {'FAKEPKG_SETTING': lambda: 'value'}
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
