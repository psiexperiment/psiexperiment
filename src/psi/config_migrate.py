'''
Convert a pre-rework ``config.py`` into ``config.toml``.

The old configuration file was executable Python whose values were often
computed (``BASE_DIRECTORY / 'data'``), so parsing it is not enough --
the values have to be evaluated to be captured. This module therefore
executes the file it is replacing, which is safe in a way the standalone
audit tool is not: it runs on a machine where psi is installed, at a
moment the user has explicitly asked for a conversion, and it is the last
time that file is ever run.

``tools/audit_legacy_config.py`` deliberately does the opposite -- it
parses with ``ast`` and never executes -- because it has to run on
machines where psi is absent or half-upgraded, and before the user has
committed to anything.
'''
import importlib.util
import logging
from pathlib import Path

from .config import _tomlify, get_config_file, save_config
from .config_legacy import (
    MERGED_INTO_SETTINGS_ROOT, REMOVED, RUNTIME_ONLY, migrate_name,
    settings_root_from_pair
)

log = logging.getLogger(__name__)


def _load_legacy_module(path):
    spec = importlib.util.spec_from_file_location('_psi_legacy_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tomlable(value):
    '''
    Convert a value from the old config into something TOML can hold.

    Paths become strings; tuples become arrays. Anything else is returned
    unchanged and validated by the caller.
    '''
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_tomlable(v) for v in value]
    if isinstance(value, list):
        return [_tomlable(v) for v in value]
    return value


def collect(source):
    '''
    Read a legacy config file and return what should be written.

    Returns
    -------
    updates : dict
        Settings to write, under their new names.
    notes : list of str
        Human-readable notes about anything not carried over.
    '''
    source = Path(source).expanduser()
    module = _load_legacy_module(source)

    updates = {}
    notes = []

    for name, value in vars(module).items():
        if name.startswith('_') or name != name.upper():
            continue
        # BASE_DIRECTORY was scaffolding in the generated template --
        # nothing read it, the other roots were written out in full. It
        # is a real setting now, and the roots that are not written out
        # derive from it, so dropping it silently sent every derived
        # setting (CFTS_ROOT, and anything added later) to the built-in
        # default instead of the rig's own tree.
        if name == 'BASE_DIRECTORY':
            updates['PSI_BASE_DIRECTORY'] = _tomlable(value)
            notes.append('BASE_DIRECTORY -> PSI_BASE_DIRECTORY (it is a '
                         'real setting now; the roots not written out '
                         'derive from it)')
            continue
        # SYSTEM is the hostname the template computed for its own use.
        if name == 'SYSTEM':
            continue
        if callable(value) or isinstance(value, type):
            continue

        if name in RUNTIME_ONLY:
            notes.append(f'{name}: dropped, {RUNTIME_ONLY[name]}')
            continue
        if name in REMOVED:
            notes.append(f'{name}: dropped, {REMOVED[name]}')
            continue

        new_name = migrate_name(name)
        if new_name is None:
            # Already current, or something local to this rig that psi
            # never read. Carry it over untouched rather than silently
            # dropping a value the user put there on purpose.
            new_name = name
            if not name.startswith(('PSI_', 'PSIDATA_', 'CFTS_', 'CFTSCAL_',
                                    'NOISE_EXP_')):
                notes.append(
                    f'{name}: carried over unchanged, but it carries no '
                    'package prefix and nothing reads it')
        else:
            notes.append(f'{name} -> {new_name}')

        updates[new_name] = _tomlable(value)

    settings_root, settings_notes = _collapse_settings_roots(vars(module))
    if settings_root is not None:
        updates['PSI_SETTINGS_ROOT'] = str(settings_root)
    notes.extend(settings_notes)

    # A package's own files win over config.py: if a rig set cftscal's
    # calibration folder in both places, the value cftscal actually used
    # was the one in its workspace.json, so it must not be clobbered by
    # the CAL_ROOT that config.py happened to carry.
    package_updates, package_notes = collect_packages(source.parent)
    updates.update(package_updates)
    notes.extend(package_notes)

    return updates, notes


def collect_packages(folder):
    '''
    Run every installed package's legacy-settings conversion.

    Some packages kept settings files of their own in the legacy
    configuration folder, beside ``config.py`` (cftscal's
    ``cfts/workspace.json``, for one). Only the package knows what those
    files hold, so each converts its own, declared by entry point::

        [project.entry-points."psi.migrations"]
        cftscal = "cftscal.migrate_settings:collect_legacy_settings"

    The entry point names a callable taking the legacy folder and
    returning ``(updates, notes)`` in the same form as `collect`.

    One package whose conversion fails is reported in the notes rather
    than raised, so it cannot stop everything else from being migrated.
    '''
    from importlib.metadata import entry_points

    updates = {}
    notes = []
    for entry in entry_points(group='psi.migrations'):
        try:
            package_updates, package_notes = entry.load()(Path(folder))
        except Exception as e:
            log.exception('Migration from %s failed', entry.name)
            notes.append(f'{entry.name}: could not convert its settings '
                         f'({e}); skipped')
            continue
        updates.update(package_updates)
        notes.extend(package_notes)
    return updates, notes


def _collapse_settings_roots(values):
    '''
    Derive PSI_SETTINGS_ROOT from a legacy layout/preferences pair.

    The old `psi-config create` wrote BASE/settings/layout and
    BASE/settings/preferences, so the pair almost always collapses to
    their common parent. When it does not -- somebody put them on
    different drives -- nothing is written and the note says so, because
    picking one is a decision this cannot make for them.
    '''
    notes = []
    for layout_key, preferences_key in MERGED_INTO_SETTINGS_ROOT:
        layout = values.get(layout_key)
        preferences = values.get(preferences_key)
        if layout is None or preferences is None:
            continue
        root = settings_root_from_pair(layout, preferences)
        if root is None:
            notes.append(
                f'{layout_key} and {preferences_key} do not sit under a '
                'common parent, so PSI_SETTINGS_ROOT could not be derived '
                'from them -- set it by hand, or move the two directories '
                'to <root>/layout and <root>/preferences')
            return None, notes
        notes.append(f'{layout_key} + {preferences_key} -> PSI_SETTINGS_ROOT')
        return root, notes
    return None, notes


def migrate(source, dry_run=False):
    '''
    Convert `source` and write the result to the current config file.
    '''
    updates, notes = collect(source)
    target = get_config_file()

    # Convert everything up front, so a dry run fails on exactly the
    # input the real run would fail on. Previously the conversion only
    # happened inside save_config, which a dry run never reached -- so
    # the dry run succeeded, and the real run printed the same complete,
    # correct-looking report and then aborted having written nothing.
    for name in sorted(updates):
        _tomlify(updates[name], name)

    print(f'Reading {Path(source).expanduser()}')
    print(f'Writing {target}')
    print()
    for note in notes:
        print(f'  {note}')
    print()
    for name in sorted(updates):
        print(f'  {name} = {updates[name]!r}')
    print()

    if dry_run:
        print('Dry run: nothing written.')
        return updates

    save_config(updates)
    print(f'Wrote {len(updates)} setting(s). Verify with `psi-config show`.')
    return updates
