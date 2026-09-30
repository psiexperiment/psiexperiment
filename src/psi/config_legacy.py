'''
Names from before the configuration rework.

Used by ``psi-config migrate`` to convert an old ``config.py`` into a
``config.toml``. Nothing else reads this: the core packages carry no
backwards compatibility, so a legacy name is simply not read any more.

``tools/audit_legacy_config.py`` carries a copy of these tables, because
it has to run on a machine where psi is not installed (or is
half-upgraded) and therefore cannot import anything from here. The copies
are kept honest by ``tests/test_config_legacy.py``, which compares them.
'''


#: Settings that were renamed. Old name -> new name.
RENAMED = {
    # psiexperiment: every setting gains the PSI_ prefix it already used
    # for its environment twin.
    'LOG_ROOT': 'PSI_LOG_ROOT',
    'DATA_ROOT': 'PSI_DATA_ROOT',
    'IO_ROOT': 'PSI_IO_ROOT',
    'HOSTNAME': 'PSI_HOSTNAME',
    'WEBSOCKETS_URI': 'PSI_WEBSOCKETS_URI',

    # Ownership move: where calibration files are stored belongs to
    # cftscal, which already read CFTSCAL_ROOT from the environment. The
    # CAL_ROOT key that `psi-config` used to emit was read by nothing.
    'CAL_ROOT': 'CFTSCAL_ROOT',

    # psidata / cftsdata read these straight from the environment with no
    # prefix at all -- the highest collision risk in the tree.
    'RAW_DATA_DIR': 'PSIDATA_RAW_DIR',
    'PROC_DATA_DIR': 'PSIDATA_PROC_DIR',
}


#: Settings that are gone entirely, as opposed to renamed.
#: Name -> explanation.
REMOVED = {
    # There is no "config folder" any more. Every directory is its own
    # named setting, and the config file is named outright.
    'PSI_CONFIG': 'the config folder variable is gone; name the file '
                  'directly with PSI_CONFIG_FILE, and set each directory '
                  'through its own setting (PSI_BASE_DIRECTORY, '
                  'PSI_DATA_ROOT, CFTS_ROOT, CFTSCAL_ROOT, ...)',

    # Emitted by the old `psi-config create` but read by nothing, in any
    # package. Processed data is addressed by PSIDATA_PROC_DIR.
    'PROCESSED_ROOT': 'was read by nothing; processed data is addressed by '
                      'PSIDATA_PROC_DIR',
    'PSI_PROCESSED_ROOT': 'was read by nothing; processed data is addressed '
                          'by PSIDATA_PROC_DIR',

    # Collapsed into PSI_SETTINGS_ROOT, which holds both as
    # <root>/layout and <root>/preferences. Handled specially by
    # psi-config migrate, which derives the single root from the pair
    # when they share a parent; see MERGED_INTO_SETTINGS_ROOT.
    'LAYOUT_ROOT': 'replaced by PSI_SETTINGS_ROOT, which holds layouts under '
                   '<root>/layout',
    'PREFERENCES_ROOT': 'replaced by PSI_SETTINGS_ROOT, which holds '
                        'preferences under <root>/preferences',
    'PSI_LAYOUT_ROOT': 'replaced by PSI_SETTINGS_ROOT, which holds layouts '
                       'under <root>/layout',
    'PSI_PREFERENCES_ROOT': 'replaced by PSI_SETTINGS_ROOT, which holds '
                            'preferences under <root>/preferences',

    # Only ever padded an error listing. list_io matched a hostname
    # against these, and they are dotted module paths, which a hostname
    # is never a substring of.
    # Written on every launch and read by nothing, in any package.
    'ARGS': 'removed; it held the parsed command line and nothing read it',

    # Carried on the workbench now (PSIWorkbench.experiment_name),
    # set from the paradigm named on the command line.
    'EXPERIMENT': 'removed; psi takes the paradigm from the command line',

    'STANDARD_IO': 'removed; psi finds IO manifests in PSI_IO_ROOT',
    'PSI_STANDARD_IO': 'removed; psi finds IO manifests in PSI_IO_ROOT',

    # get_paradigm imports the module itself when given a dotted name,
    # so `psi cfts.paradigms.abr_io` needs no configuration at all.
    'PARADIGM_DESCRIPTIONS': 'removed; name a paradigm by its full '
                             'dotted path, e.g. cfts.paradigms.abr_io',
    'PSI_PARADIGM_DESCRIPTIONS': 'removed; name a paradigm by its full '
                                 'dotted path, e.g. cfts.paradigms.abr_io',
}


#: Keys written at runtime by the application. They must never appear in
#: a config file or the environment. Name -> what sets it.
RUNTIME_ONLY = {
    'LOG_FILENAME': 'set by the psi launcher when logging is configured',
    'PROFILE': 'set by the psi launcher from --profile',
}


#: Handoff variables: written into the environment of the `psi`
#: subprocess by cftscal (and the launchers built on it) and read by the
#: paradigm manifests. The namespace moves to CFTSCAL_, because cftscal
#: owns the format and exports the manifests that read it. CFTS_ROOT is a
#: genuine cfts setting and is excluded.
HANDOFF_PREFIX_OLD = 'CFTS_'
HANDOFF_PREFIX_NEW = 'CFTSCAL_'
HANDOFF_KEEP = {'CFTS_ROOT'}


def migrate_name(name):
    '''
    New name for a legacy setting, or None if it has none.

    Returns None for names that were removed, are runtime-only, or are
    already current -- the caller decides how to report each case.
    '''
    if name in RENAMED:
        return RENAMED[name]
    if name.startswith(HANDOFF_PREFIX_OLD) and name not in HANDOFF_KEEP:
        return HANDOFF_PREFIX_NEW + name[len(HANDOFF_PREFIX_OLD):]
    return None


#: The layout/preferences pair that PSI_SETTINGS_ROOT replaced, in the
#: spellings a configuration file may carry. `psi-config migrate` derives
#: the single root from them when they sit side by side under a common
#: parent -- which is what the old `psi-config create` produced
#: (BASE/settings/layout and BASE/settings/preferences).
MERGED_INTO_SETTINGS_ROOT = (
    ('LAYOUT_ROOT', 'PREFERENCES_ROOT'),
    ('PSI_LAYOUT_ROOT', 'PSI_PREFERENCES_ROOT'),
)


def settings_root_from_pair(layout, preferences):
    '''
    The PSI_SETTINGS_ROOT that replaces a layout/preferences pair.

    Returns None when the two are not `<parent>/layout` and
    `<parent>/preferences` under the same parent, since there is then no
    single root that reproduces both and the choice belongs to whoever is
    migrating rather than to this function.
    '''
    from pathlib import Path

    layout, preferences = Path(layout), Path(preferences)
    if layout.name != 'layout' or preferences.name != 'preferences':
        return None
    if layout.parent != preferences.parent:
        return None
    return layout.parent
