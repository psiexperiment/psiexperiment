'''
The ``psi-config`` command: show, read and write psi's settings.

Kept apart from :mod:`psi.application`, which is the experiment launcher
and imports enaml; nothing here needs the GUI machinery, and a script
calling ``psi-config get`` should not pay for loading it.
'''
import argparse
import logging
from pathlib import Path

import psi
from psi import register_defaults
from psi.core.console import setup_windows_console

log = logging.getLogger(__name__)


def list_io_templates():
    '''
    The IO manifest templates `create-io` can copy, shipped with psi.

    Those whose name starts with an underscore cannot be run as they are
    and must be edited first; all of them can be used as a starting
    point.
    '''
    io_template_path = Path(__file__).parent / 'templates' / 'io'
    return list(io_template_path.glob('*.enaml'))


def _io_template_choices():
    # The name `create-io` takes for each template, underscore removed.
    return {p.stem.strip('_'): p for p in list_io_templates()}


def _render_setting(value, verbose=False):
    '''
    One line for a setting's value.

    Paths print as the path rather than as WindowsPath('...'), and
    strings without quotes: this listing is read to check a location
    or a device name, and those should be pasteable. Containers are
    summarized, because a nested table (cftscal's per-plugin state
    runs to dozens of keys) otherwise buries every other setting on
    the screen. `verbose` prints them in full.
    '''
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value if value else '(empty)'
    if isinstance(value, bool) or not isinstance(value, (list, tuple, dict)):
        return str(value)

    if verbose:
        return repr(value)
    if not value:
        return '(none)'
    if isinstance(value, dict):
        keys = ', '.join(sorted(value))
        n = len(value)
        return f'({n} {"entry" if n == 1 else "entries"}: {keys})'
    if all(isinstance(v, (str, int, float, Path)) for v in value):
        return ', '.join(str(v) for v in value)
    n = len(value)
    return f'({n} {"item" if n == 1 else "items"})'


def _register_downstream_settings():
    '''
    Register the settings of every installed package that declares some.

    A package registers its settings when it is imported, and a tool like
    `psi-config` imports only psi. Without this, every CFTSCAL_ or
    NOISE_EXP_ key in the configuration file is reported as belonging to
    no setting at all -- which is exactly backwards, since those are the
    settings somebody running `psi-config` is most likely to be checking.

    Discovery is by entry point, so it depends only on what is installed::

        [project.entry-points."psi.settings"]
        cftscal = "cftscal.config_defaults:DEFAULTS"

    Nothing here consults the configuration file, which is the point: a
    package declares its settings by being installed, not by being named
    somewhere.

    Failures are logged rather than raised. `psi-config show` is what
    somebody runs when something is already broken, so one package that
    will not import must not take the listing down with it.
    '''
    from importlib.metadata import entry_points

    loaded = []
    for entry in entry_points(group='psi.settings'):
        try:
            register_defaults(entry.load())
            loaded.append(entry.name)
        except Exception as e:
            log.warning('Could not register the settings declared by %s, so '
                        'they will be reported as unknown: %s', entry.name, e)
    return loaded


def _group_label(names):
    '''
    The prefix a group of settings shares.

    Taken from the names rather than from a hard-coded list of
    packages, since any package can register its own. Segments are
    only consumed while every name agrees and never down to a
    name's last segment, so NOISE_EXP_* is labelled NOISE_EXP
    while a lone CFTS_ROOT is labelled CFTS rather than
    CFTS_ROOT.
    '''
    parts = [n.split('_') for n in names]
    common = []
    for i in range(min(len(p) - 1 for p in parts)):
        segment = {p[i] for p in parts}
        if len(segment) != 1:
            break
        common.append(segment.pop())
    return '_'.join(common) or names[0].split('_')[0]


def _format_for_shell(name, value):
    '''
    Render a setting's value as the one line `psi-config get` prints.

    The output is meant to be captured by a script (a Windows batch file's
    ``for /f``, a shell's ``$(...)``), so it is the bare value in the same
    spelling the setting accepts back from the environment or from
    `psi-config set`: paths as plain paths, true/false for switches, lists
    joined with commas, nothing at all for an unset value.

    Raises
    ------
    ValueError
        For a table (or a list holding one), which has no one-line form a
        script could use.
    '''
    if value is None:
        return ''
    # bool before anything numeric: bool is a subclass of int.
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, dict):
        raise ValueError(
            f'{name} holds a table of values, which has no one-line form. '
            'Run `psi-config show --verbose` to see it.')
    if isinstance(value, (list, tuple)):
        if any(isinstance(v, (dict, list, tuple)) for v in value):
            raise ValueError(
                f'{name} holds nested values, which have no one-line form. '
                'Run `psi-config show --verbose` to see it.')
        return ','.join(_format_for_shell(name, v) for v in value)
    return str(value)


def show_config(args):
    loaded = _register_downstream_settings()
    config_file = psi.get_config_file()
    exists = 'exists' if config_file.exists() else 'does not exist'
    print(f'Configuration file: {config_file} ({exists})')
    if loaded:
        print(f'Also loaded: {", ".join(sorted(loaded))}')

    settings = psi.get_all_config()
    if not settings:
        print('\nNo settings are known.')
        return

    # A key in the file that no package registered is read by nothing.
    # Listing it beside the real settings implies it does something.
    known = {n: v for n, v in settings.items()
             if psi.get_setting(n) is not None}
    unknown = {n: v for n, v in settings.items() if n not in known}

    # The source is the whole point of this listing -- "why is this not
    # what I put in the file?" -- but most settings are defaults, and
    # labelling every one of them just crowds out the few that matter.
    labels = {'config file': 'file', 'environment': 'env', 'default': ''}
    width = max(len(n) for n in settings)

    groups = {}
    for name, value in known.items():
        groups.setdefault(name.split('_', 1)[0], []).append((name, value))

    rendered = {n: _render_setting(v, args.verbose)
                for n, v in known.items()}
    # Wide enough for most values, but not so wide that one long entry
    # pushes the source column off the screen for everything else.
    vwidth = min(max((len(v) for v in rendered.values()), default=0), 44)

    for key in sorted(groups):
        entries = sorted(groups[key])
        print(f'\n{_group_label([n for n, _ in entries])}')
        for name, _ in entries:
            source = labels[psi.config_source(name)]
            print(f'  {name:<{width}}  {rendered[name]:<{vwidth}}  '
                  f'{source}'.rstrip())
            doc = psi.get_setting(name).doc
            if args.verbose and doc:
                print(f'  {"":<{width}}  {doc}')

    if unknown:
        print('\nIn the configuration file but not registered by any '
              'package loaded here')
        for name, value in sorted(unknown.items()):
            print(f'  {name:<{width}}  '
                  f'{_render_setting(value, args.verbose)}')
        print('  (left over from an older version, misspelled, or owned '
              'by a package that is')
        print('   not installed here)')

    print('\nA blank source means the package default.')
    if not args.verbose:
        print('Run with --verbose to print container values in full.')

def get_config_value(args):
    # Registered first for the same reason as show and set: without
    # the owning package's defaults, a CFTSCAL_ or NOISE_EXP_ setting
    # that is not in the file is unknown, and one that is comes back
    # uncoerced (a path as a plain string, a number as text).
    _register_downstream_settings()
    try:
        value = psi.get_config(args.setting)
    except KeyError:
        # The message goes to stderr and the exit status is non-zero, so
        # a script that captures stdout gets nothing rather than an
        # error message it would take for the value.
        raise SystemExit(
            f'{args.setting} is not a known setting and is not in the '
            f'configuration file ({psi.get_config_file()}).') from None
    try:
        print(_format_for_shell(args.setting, value))
    except ValueError as e:
        raise SystemExit(str(e)) from None

def set_config_value(args):
    # Same reason as show: without the owning package loaded, a
    # downstream setting has no registered default, so the value would
    # not be coerced to its type and config_source could not report
    # that the environment is shadowing the write.
    _register_downstream_settings()

    # A misspelled name would otherwise be written as a key that
    # nothing reads, and reported as a success.
    if psi.get_setting(args.setting) is None and not args.force:
        import difflib
        close = difflib.get_close_matches(
            args.setting, psi.setting_names(), n=1)
        hint = f' Did you mean {close[0]}?' if close else ''
        raise SystemExit(
            f'{args.setting} is not a setting any installed package '
            f'declares.{hint} Run `psi-config show` for the known '
            'settings, or pass --force to write it anyway.')

    # A command-line value is a string, and save_config replaces the
    # key outright. For a setting that holds a table -- cftscal keeps
    # every plugin's saved state in one -- that silently destroys it,
    # and the next launch fails reading a str where a dict belongs.
    if psi.setting_type(args.setting) is dict:
        raise SystemExit(
            f'{args.setting} holds a table of values, which cannot be '
            'set from the command line -- doing so would replace the '
            'whole table. Edit the configuration file directly, or let '
            'the application that owns this setting write it.')

    # Validate before writing, not after. Writing first and reading
    # back second left a rejected value in the file -- and for a path
    # setting every later read then raises, so the rig will not start
    # and the only way out is hand-editing TOML. The converted value
    # is what gets written, so a number, switch or list goes into the
    # file as a real TOML number, boolean or array rather than as a
    # string that only reads back correctly because it is parsed.
    try:
        value = psi.parse_setting(args.setting, args.value)
    except ValueError as e:
        # The message says everything; the ValueError's traceback
        # would only bury it.
        raise SystemExit(f'{args.setting}: {e}') from None
    if isinstance(value, Path):
        # Checked, but written as typed: Path() would turn D:/data
        # into D:\data in a file the user edits by hand.
        value = args.value

    psi.save_config({args.setting: value})
    source = psi.config_source(args.setting)
    value = psi.get_config(args.setting)
    if isinstance(value, Path):
        value = str(value)
    print(f'{args.setting} = {value}')
    if source == 'environment':
        # Writing succeeded but changed nothing the application will
        # see. Saying so here avoids a long hunt later.
        print(f'WARNING: {args.setting} is also set in the environment, '
              'which takes precedence. The value written to the '
              'configuration file will not take effect until the '
              'environment variable is cleared.')

def migrate_config(args):
    from psi.config_migrate import migrate
    migrate(args.source, dry_run=args.dry_run)

def create_config(args):
    # create_config_dirs walks the registered settings, so without the
    # downstream packages loaded it makes only psi's own directories,
    # silently skips CFTSCAL_ROOT and every other root a package
    # declares -- on the one command whose whole job is laying out
    # the tree for a new rig.
    _register_downstream_settings()
    base_directory = args.base_directory.rstrip('\\')
    psi.create_config(base_directory=base_directory)
    if args.base_directory:
        psi.create_config_dirs()

def create_folders(args):
    _register_downstream_settings()
    psi.create_config_dirs()

def create_io(args):
    template = _io_template_choices()[args.template].name
    psi.create_io_manifest(template)


def build_parser():
    parser = argparse.ArgumentParser(
        'psi-config',
        description='Configure psiexperiment'
    )
    subparsers = parser.add_subparsers(
        dest='cmd',
        description='Available actions',
    )
    subparsers.required = True

    show = subparsers.add_parser(
        'show',
        description='Show the config file location, every setting, its '
                    'resolved value and which layer supplied it.',
    )
    show.set_defaults(func=show_config)
    show.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print container values in full instead of summarizing them.',
    )

    get_parser = subparsers.add_parser(
        'get',
        description='Print the resolved value of a setting, and nothing '
                    'else, for use in scripts. In a Windows batch file: '
                    'for /f "usebackq delims=" %%i in '
                    '(`psi-config get PSI_DATA_ROOT`) do set "DATA_ROOT=%%i"',
    )
    get_parser.set_defaults(func=get_config_value)
    get_parser.add_argument('setting', type=str, help='Name of the setting.')

    set_parser = subparsers.add_parser(
        'set',
        description='Write a setting to the configuration file.',
    )
    set_parser.set_defaults(func=set_config_value)
    set_parser.add_argument('setting', type=str, help='Name of the setting.')
    set_parser.add_argument('value', type=str, help='Value to write.')
    set_parser.add_argument(
        '--force',
        action='store_true',
        help='Write the setting even though no installed package declares '
             'it (for one owned by a package not installed here).',
    )

    migrate = subparsers.add_parser(
        'migrate',
        description='Convert a pre-rework configuration into config.toml: '
                    'the legacy config.py, if there is one, and the '
                    'settings files installed packages kept beside it '
                    '(such as the workspace.json cftscal kept).',
    )
    migrate.set_defaults(func=migrate_config)
    migrate.add_argument(
        'source',
        type=Path,
        nargs='?',
        default=None,
        help='The legacy config.py, or the folder holding the legacy files '
             '(default: $PSI_CONFIG, or ~/psi). The folder need not have a '
             'config.py.',
    )
    migrate.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be written without writing it.',
    )

    create = subparsers.add_parser('create')
    create.set_defaults(func=create_config)
    create.add_argument(
        '--base-directory',
        type=str,
        help='Root directory to store data and settings for psiexperiment.'
    )

    make = subparsers.add_parser(
        'create-folders',
        description='Create folders defined in the config file.',
    )
    make.set_defaults(func=create_folders)

    io = subparsers.add_parser(
        'create-io',
        description='Creates a hardware configuration skeleton that you can edit'
    )
    io.set_defaults(func=create_io)
    io.add_argument(
        'template',
        type=str,
        choices=sorted(_io_template_choices()),
        help='Template to use for hardware configuration skeleton.',
    )

    return parser


def main():
    setup_windows_console()
    args = build_parser().parse_args()
    args.func(args)
