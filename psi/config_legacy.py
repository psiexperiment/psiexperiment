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
    'PROCESSED_ROOT': 'PSI_PROCESSED_ROOT',
    'PREFERENCES_ROOT': 'PSI_PREFERENCES_ROOT',
    'LAYOUT_ROOT': 'PSI_LAYOUT_ROOT',
    'IO_ROOT': 'PSI_IO_ROOT',
    'HOSTNAME': 'PSI_HOSTNAME',
    'STANDARD_IO': 'PSI_STANDARD_IO',
    'PARADIGM_DESCRIPTIONS': 'PSI_PARADIGM_DESCRIPTIONS',
    'WEBSOCKETS_URI': 'PSI_WEBSOCKETS_URI',

    # psiexperiment IO templates.
    'NI_EEG_CHANNEL': 'PSI_NI_EEG_CHANNEL',
    'NI_CALIBRATION_CHANNEL': 'PSI_NI_CALIBRATION_CHANNEL',
    'NI_STARSHIP_CHANNEL': 'PSI_NI_STARSHIP_CHANNEL',
    'NI_START_TRIGGER': 'PSI_NI_START_TRIGGER',
    # The key was misspelled in PXIe-1062.enaml, so a rig that worked
    # around it by spelling the key wrong too must also be fixed.
    'NI_START_TRIGER': 'PSI_NI_START_TRIGGER',

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
}


#: Keys written at runtime by the application. They must never appear in
#: a config file or the environment. Name -> what sets it.
RUNTIME_ONLY = {
    'EXPERIMENT': 'set by the psi launcher from the command line',
    'LOG_FILENAME': 'set by the psi launcher when logging is configured',
    'ARGS': 'set by the psi launcher; holds parsed arguments',
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
