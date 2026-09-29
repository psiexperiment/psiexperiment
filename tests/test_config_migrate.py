'''
End-to-end tests for `psi-config migrate`.

This is the one-shot, irreversible, per-rig upgrade path, and it had no
tests at all. Two bugs lived in it as a result, both of which reported
success:

- the cftscal per-plugin tables were keyed by the filename *stem* while
  cftscal reads them by the full filename, so every plugin silently
  reverted to defaults -- and MIGRATION.md then told the user to delete
  the source files;
- a None anywhere inside a collected value aborted the write with a raw
  tomlkit traceback, after the full and correct-looking report had
  already printed.

The fixtures below are shaped like what `psi-config create` actually
produced, since that is what a rig is migrating from.
'''
import json
from pathlib import Path

import pytest

from psi import config as psi_config
from psi import get_config
from psi.config_migrate import collect, migrate


LEGACY_CONFIG = '''
from pathlib import Path
import socket

SYSTEM = socket.gethostname()
BASE_DIRECTORY = Path(r'C:\\Data\\psi')

LOG_ROOT = BASE_DIRECTORY / 'logs'
DATA_ROOT = BASE_DIRECTORY / 'data'
PROCESSED_ROOT = BASE_DIRECTORY / 'processed'
CAL_ROOT = BASE_DIRECTORY / 'calibration'
PREFERENCES_ROOT = BASE_DIRECTORY / 'settings' / 'preferences'
LAYOUT_ROOT = BASE_DIRECTORY / 'settings' / 'layout'
IO_ROOT = BASE_DIRECTORY / 'io'
'''


@pytest.fixture
def rig(tmp_path, monkeypatch):
    '''
    A legacy configuration directory, plus the file psi will migrate into.
    '''
    source = tmp_path / 'config.py'
    source.write_text(LEGACY_CONFIG, encoding='utf-8')
    monkeypatch.setenv('PSI_CONFIG_FILE', str(tmp_path / 'config.toml'))
    psi_config.reload_config()
    yield source
    psi_config.reload_config()


def write_plugin(source, name, data):
    path = source.parent / 'cfts' / 'calibration' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def write_workspace(source, data):
    path = source.parent / 'cfts' / 'workspace.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


class TestPluginTables:

    def test_keyed_by_the_name_cftscal_reads(self, rig):
        '''
        CalibrationSettings.settings_filename is 'microphone.json', and it
        is used verbatim as the key. Keying by the stem wrote a table
        nothing read.
        '''
        write_plugin(rig, 'microphone-measurement.json', {'gain': 20})
        updates, _ = collect(rig)
        assert 'microphone-measurement.json' in updates['CFTSCAL_PLUGIN']
        assert 'microphone-measurement' not in updates['CFTSCAL_PLUGIN']

    def test_round_trips_to_what_cftscal_looks_up(self, rig):
        write_plugin(rig, 'speaker.json', {'output': 'ao0'})
        migrate(rig)
        # The exact lookup in CalibrationSettings.load_config. A plain
        # dict either way -- CFTSCAL_PLUGIN has no registered default
        # here, but a table is a table.
        table = get_config('CFTSCAL_PLUGIN')
        assert table.get('speaker.json') == {'output': 'ao0'}

    def test_every_plugin_survives(self, rig):
        for name in ('microphone-measurement.json', 'speaker.json',
                     'starship.json'):
            write_plugin(rig, name, {'name': name})
        migrate(rig)
        assert set(get_config('CFTSCAL_PLUGIN')) == {
            'microphone-measurement.json', 'speaker.json', 'starship.json'}

    def test_unreadable_plugin_is_skipped_not_fatal(self, rig):
        write_plugin(rig, 'good.json', {'a': 1})
        bad = rig.parent / 'cfts' / 'calibration' / 'bad.json'
        bad.write_text('{not json', encoding='utf-8')
        updates, notes = collect(rig)
        assert 'good.json' in updates['CFTSCAL_PLUGIN']
        assert any('bad.json' in n and 'skipped' in n for n in notes)


class TestNullValues:

    def test_none_inside_a_table_is_dropped(self, rig):
        '''
        A launcher writes this on a fresh install, where no device has
        been chosen yet. It used to abort the whole migration.
        '''
        write_workspace(rig, {'data_path': 'C:/cal', 'sample_rate': None})
        # Asserted on the mapping rather than the resolved value: these
        # are cftscal's settings, and psiexperiment's tests do not import
        # cftscal, so nothing has registered a default to coerce against.
        updates, _ = collect(rig)
        assert updates['CFTSCAL_ROOT'] == 'C:/cal'
        # collect reports what the old file held, including the None;
        # dropping it is the writer's job.
        assert updates['CFTSCAL_SAMPLE_RATE'] is None

        migrate(rig)
        assert 'CFTSCAL_SAMPLE_RATE' not in psi_config.load_config()

    def test_none_in_a_plugin_table_is_dropped(self, rig):
        write_plugin(rig, 'microphone.json', {'gain': 20, 'device': None})
        migrate(rig)
        assert get_config('CFTSCAL_PLUGIN')['microphone.json'] == {'gain': 20}


class TestSettingsRoots:

    def test_layout_and_preferences_collapse(self, rig):
        migrate(rig)
        assert get_config('PSI_SETTINGS_ROOT') == Path(r'C:\Data\psi\settings')

    def test_dead_settings_are_dropped(self, rig):
        updates, notes = collect(rig)
        for name in ('PROCESSED_ROOT', 'PSI_PROCESSED_ROOT',
                     'PSI_LAYOUT_ROOT', 'PSI_PREFERENCES_ROOT'):
            assert name not in updates
        assert any('PROCESSED_ROOT' in n and 'dropped' in n for n in notes)

    def test_surviving_roots_keep_their_paths(self, rig):
        migrate(rig)
        assert get_config('PSI_DATA_ROOT') == Path(r'C:\Data\psi\data')
        assert get_config('PSI_IO_ROOT') == Path(r'C:\Data\psi\io')
        # Migrated rigs keep the log location they had, even though the
        # default no longer derives from the base directory.
        assert get_config('PSI_LOG_ROOT') == Path(r'C:\Data\psi\logs')


class TestWorkspace:

    def test_workspace_wins_over_cal_root(self, rig):
        '''
        Both name the calibration folder, and workspace.json is the one
        cftscal actually used, so it must not be clobbered by CAL_ROOT.
        '''
        write_workspace(rig, {'data_path': 'C:/Calibration/real'})
        updates, _ = collect(rig)
        assert updates['CFTSCAL_ROOT'] == 'C:/Calibration/real'

    def test_device_identity_is_carried_over(self, rig):
        write_workspace(rig, {'selected_device_name': 'Fireface',
                              'selected_device_hostapi': 'ASIO',
                              'sample_rate': 96000})
        updates, _ = collect(rig)
        assert updates['CFTSCAL_DEVICE_NAME'] == 'Fireface'
        assert updates['CFTSCAL_DEVICE_HOSTAPI'] == 'ASIO'
        assert updates['CFTSCAL_SAMPLE_RATE'] == 96000

    def test_unreadable_workspace_is_not_fatal(self, rig):
        '''
        A truncated workspace.json must not abort the conversion of
        everything else -- the plugin reader already behaves this way.
        '''
        path = rig.parent / 'cfts' / 'workspace.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{not json', encoding='utf-8')
        updates, notes = collect(rig)
        assert updates['PSI_DATA_ROOT'] == r'C:\Data\psi\data'
        assert any('workspace.json' in n for n in notes)


class TestDryRun:

    def test_dry_run_writes_nothing(self, rig, tmp_path):
        migrate(rig, dry_run=True)
        assert not (tmp_path / 'config.toml').exists()

    def test_dry_run_exercises_the_same_conversion(self, rig):
        '''
        A dry run that stops short of the conversion cannot warn about an
        input the real run chokes on, which is exactly what happened with
        a None in a table.
        '''
        # A value the writer will reject, not merely one it transforms:
        # comparing two collect() dicts passed while the property was
        # false, because collect is not where the conversion happens.
        write_workspace(rig, {'enabled_plugins': ['microphone', None]})
        with pytest.raises(ValueError, match='element 1 is None'):
            migrate(rig, dry_run=True)
        with pytest.raises(ValueError, match='element 1 is None'):
            migrate(rig)
