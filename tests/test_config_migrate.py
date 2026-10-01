'''
End-to-end tests for `psi-config migrate`.

This is the one-shot, irreversible, per-rig upgrade path, and it had no
tests at all. Two bugs lived in it as a result, both of which reported
success:

- the cftscal per-plugin tables were keyed by the filename *stem* while
  cftscal reads them by the full filename, so every plugin silently
  reverted to defaults -- and MIGRATION.md then told the user to delete
  the source files (cftscal converts its own files now, and tests that
  conversion itself);
- a None anywhere inside a collected value aborted the write with a raw
  tomlkit traceback, after the full and correct-looking report had
  already printed.

The fixtures below are shaped like what `psi-config create` actually
produced, since that is what a rig is migrating from.
'''
from pathlib import Path
from types import SimpleNamespace

import pytest

from psi import config as psi_config
from psi import get_config
from psi.config_migrate import (
    collect, default_legacy_folder, migrate, resolve_source
)


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


def package_migrations(monkeypatch, **migrations):
    '''
    Stand in for the installed packages' ``psi.migrations`` entry points.
    '''
    import importlib.metadata

    entries = [SimpleNamespace(name=name, load=lambda fn=fn: fn)
               for name, fn in migrations.items()]
    monkeypatch.setattr(importlib.metadata, 'entry_points',
                        lambda group=None: entries)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    '''
    A legacy configuration directory, plus the file psi will migrate into.

    No package migrations are installed unless a test adds them, so these
    tests do not depend on what else happens to be installed.
    '''
    source = tmp_path / 'config.py'
    source.write_text(LEGACY_CONFIG, encoding='utf-8')
    monkeypatch.setenv('PSI_CONFIG_FILE', str(tmp_path / 'config.toml'))
    package_migrations(monkeypatch)
    psi_config.reload_config()
    yield source
    psi_config.reload_config()


def add_to_config(source, text):
    source.write_text(LEGACY_CONFIG + text, encoding='utf-8')


class TestNullValues:

    def test_none_inside_a_table_is_dropped(self, rig):
        '''
        A launcher writes this on a fresh install, where no device has
        been chosen yet. It used to abort the whole migration.
        '''
        add_to_config(rig, "PSI_T_TABLE = {'device': 'a', 'gain': None}\n")
        # collect reports what the old file held, including the None;
        # dropping it is the writer's job.
        updates, _ = collect(rig)
        assert updates['PSI_T_TABLE'] == {'device': 'a', 'gain': None}

        migrate(rig)
        assert psi_config.load_config()['PSI_T_TABLE'] == {'device': 'a'}


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


class TestPackageMigrations:
    '''
    Packages convert the settings files they kept in the legacy folder
    themselves, through the ``psi.migrations`` entry point.
    '''

    def test_package_gets_the_legacy_folder(self, rig, monkeypatch):
        seen = []

        def convert(folder):
            seen.append(folder)
            return {'FAKEPKG_ROOT': 'C:/fake'}, ['fakepkg: converted']

        package_migrations(monkeypatch, fakepkg=convert)
        updates, notes = collect(rig)
        assert seen == [rig.parent]
        assert updates['FAKEPKG_ROOT'] == 'C:/fake'
        assert 'fakepkg: converted' in notes

    def test_package_wins_over_config_py(self, rig, monkeypatch):
        # config.py's CAL_ROOT becomes CFTSCAL_ROOT, but the folder cftscal
        # actually used was the one in its own files.
        package_migrations(monkeypatch, cftscal=lambda folder: (
            {'CFTSCAL_ROOT': 'C:/Calibration/real'}, []))
        updates, _ = collect(rig)
        assert updates['CFTSCAL_ROOT'] == 'C:/Calibration/real'

    def test_failing_package_is_a_note_not_fatal(self, rig, monkeypatch):
        def broken(folder):
            raise RuntimeError('unreadable')

        package_migrations(
            monkeypatch, broken=broken,
            fakepkg=lambda folder: ({'FAKEPKG_ROOT': 'C:/fake'}, []))
        updates, notes = collect(rig)
        assert updates['FAKEPKG_ROOT'] == 'C:/fake'
        assert updates['PSI_DATA_ROOT'] == r'C:\Data\psi\data'
        assert any('broken' in n and 'unreadable' in n for n in notes)


class TestWithoutConfigPy:
    '''
    A machine that only ever ran a package with settings files of its own
    (cftscal, for one) has those files in the legacy folder and no
    config.py at all.
    '''

    @pytest.fixture
    def folder(self, tmp_path, monkeypatch):
        folder = tmp_path / 'legacy'
        folder.mkdir()
        monkeypatch.setenv('PSI_CONFIG', str(folder))
        monkeypatch.setenv('PSI_CONFIG_FILE', str(tmp_path / 'config.toml'))
        psi_config.reload_config()
        yield folder
        psi_config.reload_config()

    def test_default_folder_follows_psi_config(self, folder):
        assert default_legacy_folder() == folder

    def test_default_folder_is_home(self, monkeypatch):
        monkeypatch.delenv('PSI_CONFIG', raising=False)
        assert default_legacy_folder() == Path('~/psi').expanduser()

    def test_resolve_folder_without_config_py(self, folder):
        assert resolve_source(folder) == (folder, None)
        assert resolve_source() == (folder, None)

    def test_resolve_folder_with_config_py(self, folder):
        config_py = folder / 'config.py'
        config_py.write_text(LEGACY_CONFIG, encoding='utf-8')
        assert resolve_source(folder) == (folder, config_py)
        assert resolve_source() == (folder, config_py)
        assert resolve_source(config_py) == (folder, config_py)

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_source(tmp_path / 'nowhere' / 'config.py')

    @pytest.mark.parametrize('source', ['folder', None])
    def test_packages_are_migrated(self, folder, monkeypatch, tmp_path,
                                   source):
        seen = []

        def convert(f):
            seen.append(f)
            return {'CFTSCAL_ROOT': 'C:/Calibration'}, ['cftscal: converted']

        package_migrations(monkeypatch, cftscal=convert)
        migrate(folder if source == 'folder' else None)
        assert seen == [folder]
        assert psi_config.load_config()['CFTSCAL_ROOT'] == 'C:/Calibration'

    def test_nothing_to_migrate_writes_nothing(self, folder, monkeypatch,
                                               tmp_path, capsys):
        package_migrations(monkeypatch, cftscal=lambda f: ({}, []))
        assert migrate() == {}
        assert not (tmp_path / 'config.toml').exists()
        assert 'Nothing to migrate' in capsys.readouterr().out


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
        add_to_config(rig, "PSI_T_LIST = ['microphone', None]\n")
        with pytest.raises(ValueError, match='element 1 is None'):
            migrate(rig, dry_run=True)
        with pytest.raises(ValueError, match='element 1 is None'):
            migrate(rig)
