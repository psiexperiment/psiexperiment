'''
Keep the legacy name tables in sync.

`psi.config_legacy` is used by `psi-config migrate`. `tools/audit_legacy_config.py`
carries a copy, because it has to run on machines where psi is absent or
half-upgraded and therefore cannot import from the package. Two copies of
a rename table is a maintenance hazard unless something compares them.
'''
import importlib.util
from pathlib import Path

import pytest

from psi import config_legacy


AUDIT_SCRIPT = Path(__file__).parent.parent / 'tools' / 'audit_legacy_config.py'


@pytest.fixture(scope='module')
def audit():
    '''
    Import the standalone audit script by path.

    Safe to import: it is stdlib-only and has no import-time side
    effects beyond defining its tables.
    '''
    spec = importlib.util.spec_from_file_location('_audit', AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_script_exists():
    assert AUDIT_SCRIPT.exists()


@pytest.mark.parametrize('table', ['RENAMED', 'REMOVED', 'RUNTIME_ONLY'])
def test_tables_match(audit, table):
    assert getattr(audit, table) == getattr(config_legacy, table)


def test_handoff_prefixes_match(audit):
    assert audit.HANDOFF_PREFIX_OLD == config_legacy.HANDOFF_PREFIX_OLD
    assert audit.HANDOFF_PREFIX_NEW == config_legacy.HANDOFF_PREFIX_NEW
    assert audit.HANDOFF_KEEP == config_legacy.HANDOFF_KEEP


def test_cfts_root_is_not_renamed():
    '''
    CFTS_ROOT stays: it is where the cfts launcher keeps its saved
    experiment and hardware presets, not a cftscal handoff variable.
    '''
    assert config_legacy.migrate_name('CFTS_ROOT') is None
    assert 'CFTS_ROOT' not in config_legacy.RENAMED


def test_handoff_variables_are_renamed():
    assert config_legacy.migrate_name('CFTS_MICROPHONE') == \
        'CFTSCAL_MICROPHONE'
    assert config_legacy.migrate_name('CFTS_MICROPHONE_MIC_A_GAIN') == \
        'CFTSCAL_MICROPHONE_MIC_A_GAIN'


def test_renamed_targets_are_prefixed():
    prefixes = ('PSI_', 'PSIDATA_', 'CFTS_', 'CFTSCAL_', 'NOISE_EXP_')
    for old, new in config_legacy.RENAMED.items():
        assert new.startswith(prefixes), f'{old} -> {new} has no prefix'


def test_no_pxie_1062_settings_remain():
    '''
    The PXIe-1062 template's NI_* channel settings were never used by any
    rig, so they are simply gone -- not renamed, and not worth a migration
    note that would only invite someone to look for them.
    '''
    leftovers = [n for n in {**config_legacy.RENAMED, **config_legacy.REMOVED}
                 if n.startswith('NI_')]
    assert leftovers == []


def test_processed_root_is_removed_not_renamed():
    '''
    PSI_PROCESSED_ROOT was emitted by the old `psi-config create` and
    read by nothing in any package. Processed data is addressed by
    PSIDATA_PROC_DIR.
    '''
    for name in ('PROCESSED_ROOT', 'PSI_PROCESSED_ROOT'):
        assert name in config_legacy.REMOVED
        assert name not in config_legacy.RENAMED


@pytest.mark.parametrize('name', [
    'LAYOUT_ROOT', 'PREFERENCES_ROOT',
    'PSI_LAYOUT_ROOT', 'PSI_PREFERENCES_ROOT',
])
def test_settings_roots_are_merged_not_renamed(name):
    assert name in config_legacy.REMOVED
    assert name not in config_legacy.RENAMED


class TestSettingsRootFromPair:

    def test_common_parent(self):
        # What the old `psi-config create` produced.
        assert config_legacy.settings_root_from_pair(
            'C:/rig/settings/layout',
            'C:/rig/settings/preferences') == Path('C:/rig/settings')

    def test_different_parents_cannot_be_merged(self):
        # Two directories on different drives have no single root, and
        # picking one is not this function's decision to make.
        assert config_legacy.settings_root_from_pair(
            'C:/rig/layout', 'D:/elsewhere/preferences') is None

    def test_unexpected_leaf_names_are_not_merged(self):
        assert config_legacy.settings_root_from_pair(
            'C:/rig/lay', 'C:/rig/preferences') is None
