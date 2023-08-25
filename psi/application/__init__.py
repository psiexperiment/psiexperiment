import logging.config
log = logging.getLogger(__name__)

import datetime as dt
from glob import glob
import importlib
import os.path
from pathlib import Path
import pdb
import re
import sys
import textwrap
import traceback
import warnings

import enaml
from enaml.application import deferred_call
with enaml.imports():
    from enaml.stdlib.message_box import critical

from psi import get_config, set_config
from psi.core.enaml.api import load_manifest, load_manifest_from_file


def disable_quick_edit():
    # From https://stackoverflow.com/questions/73486528/python-script-pausing-in-cmd
    import win32console as con
    import signal

    # Missing constants in pywin
    ENABLE_EXTENDED_FLAGS = 0x0080
    ENABLE_QUICK_EDIT_MODE = 0x0040

    # Modify console mode to disable quick edit mode
    h = con.GetStdHandle(con.STD_INPUT_HANDLE)
    oldMode = h.GetConsoleMode()
    h.SetConsoleMode((oldMode | ENABLE_EXTENDED_FLAGS) &
            ~ENABLE_QUICK_EDIT_MODE)


# This ensures that running scripts from the command line does not accidentally
# pause the script.
if os.name == 'nt':
    try:
        disable_quick_edit()
    except:
        pass

mesg_template = '''
A critical exception has occurred. While we do our best to prevent these
issues, they sometimes happen. We are now attempting to shut down the program
gracefully so acquired data can be saved. Please notify the developers.

{}

{}
'''


class ExceptionHandler:
    '''
    This provides a custom exception handler. Since, during the course of
    custom exception handling, new exceptions may be raised, we temporarily
    restore the default exception handling on __enter__ and then restore the
    custom exception handler on __exit__. By using the context manager, we make
    sure that __exit__ always gets called properly.
    '''

    def __init__(self):
        self.workbench = None
        self.logfile = None

    def __enter__(self):
        sys.excepthook = sys.__excepthook__

    def __exit__(self, exc_type, exc_value, exc_tb):
        sys.excepthook = self

    def format_exception(self, args):
        if self.logfile is not None:
            log_mesg = f'The log file has been saved to {self.logfile}'
        else:
            log_mesg = 'Unfortunately, no log file was saved.'

        err_mesg = f'The error message is:\n{args[1]}'
        if args[1].__cause__ is not None:
            err_mesg = f'{err_mesg}\n\nThe above error was caused by ' \
                        f'the following error:\n{args[1].__cause__}'

        mesg = mesg_template.format(args[1], log_mesg)
        mesg = re.sub(r'(?<!\n)\n(?!\n)', ' ', mesg)
        mesg = re.sub(r' +', ' ', mesg)
        return textwrap.fill(mesg, replace_whitespace=False)

    def __call__(self, *args):
        with self:
            log.exception("Uncaught exception", exc_info=args)
            mesg = self.format_exception(args)

            if self.workbench is not None:
                core = self.workbench.get_plugin('enaml.workbench.core')
                parameters = {'stop_reason': 'error', 'skip_errors': True,
                              'error_message': mesg}
                try:
                    core.invoke_command('psi.controller.stop', parameters)
                except Exception as e:
                    log.exception(e)
                    window = self.workbench.get_plugin('enaml.workbench.ui').window
                    deferred_call(critical, window, 'Oops :(', mesg)
            sys.excepthook(*args)


exception_handler = ExceptionHandler()
sys.excepthook = exception_handler


def configure_logging(level_console=None, level_file=None, filename=None,
                      debug_exclude=None):

    logging.captureWarnings(True)
    log = logging.getLogger()

    if level_file is None and level_console is None:
        return
    elif level_file is None:
        log.setLevel(level_console)
    elif level_console is None:
        log.setLevel(level_file)
    else:
        min_level = min(getattr(logging, level_console), getattr(logging, level_file))
        log.setLevel(min_level)

    fmt = '{asctime} {levelname:10s}: {threadName:11s} - {name:40s}:: {message}'
    if level_console is not None:
        try:
            level_styles = {
                'trace': dict(color='cyan'),
                'debug': dict(color='green'),
                'info': dict(color='white'),
                'warning': dict(color='yellow'),
                'error': dict(color='magenta'),
                'critical': dict(color='red'),
            }
            import coloredlogs
            import humanfriendly
            humanfriendly.terminal.enable_ansi_support()
            formatter = coloredlogs.ColoredFormatter(fmt, style='{',
                                                     level_styles=level_styles)
        except ImportError as e:
            formatter = logging.Formatter(fmt, style='{')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level_console)
        log.addHandler(stream_handler)

    if filename is not None and level_file is not None:
        file_handler = logging.FileHandler(filename, 'w', 'UTF-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level_file)
        log.addHandler(file_handler)
        exception_handler.logfile = filename

    if debug_exclude is not None:
        for name in debug_exclude:
            logging.getLogger(name).setLevel('CRITICAL')

    tdt_logger = logging.getLogger('tdt')
    tdt_logger.setLevel('INFO')
    sys.excepthook = exception_handler


def warn_with_traceback(message, category, filename, lineno, file=None,
                        line=None):
    log = file if hasattr(file,'write') else sys.stderr
    m = warnings.formatwarning(message, category, filename, lineno, line)
    log.write(m)


def _main(args):
    set_config('EXPERIMENT', args.experiment)

    if args.debug:
        # Show debugging information. This includes full tracebacks for
        # warnings.
        dt_string = dt.datetime.now().strftime('%Y-%m-%d %H%M')
        filename = '{} {}'.format(dt_string, args.experiment)
        log_root = get_config('LOG_ROOT')
        configure_logging(args.debug_level_console,
                          args.debug_level_file,
                          os.path.join(log_root, filename),
                          args.debug_exclude)

        log.debug('Logging configured')
        log.info('Logging information captured in {}'.format(filename))
        log.info('Python executable: {}'.format(sys.executable))
        if args.debug_warning:
            warnings.showwarning = warn_with_traceback

    from psi.experiment.workbench import PSIWorkbench

    workbench = PSIWorkbench()
    plugins = [p.manifest for p in args.controller.plugins \
               if p.selected or p.required]
    workbench.register_core_plugins(args.io, plugins)

    if args.pathname is None:
        log.warn('All data will be destroyed at end of experiment')

    exception_handler.workbench = workbench
    workbench.start_workspace(args.experiment,
                              args.pathname,
                              commands=args.commands,
                              load_preferences=not args.no_preferences,
                              load_layout=not args.no_layout,
                              preferences_file=args.preferences,
                              layout_file=args.layout,
                              calibration_file=args.calibration)


def list_preferences(experiment, include_default=False):
    from psi.experiment.util import PREFERENCES_WILDCARD
    if not isinstance(experiment, str):
        experiment = experiment.name
    p_root = get_config('PREFERENCES_ROOT') / experiment
    p_glob = PREFERENCES_WILDCARD[:-1].split('(')[1]
    matches = p_root.glob(p_glob)
    if not include_default:
        matches = [p for p in matches if not p.stem == 'default']
    return sorted(Path(p) for p in matches)


def list_io():
    result = []
    if (io_path := get_config('IO_ROOT', None)) is not None:
        result.extend(io_path.glob('*.enaml'))
    result.extend(get_config('STANDARD_IO', []))
    return result


def get_calibration_path(io_file):
    io_file = Path(io_file)
    return io_file.parent / io_file.stem


def get_calibration_file(io_file=None, name='default'):
    if io_file is None:
        io_file = get_default_io()
    path = get_calibration_path(io_file)
    return (path / name).with_suffix('.json')


def list_calibrations(io_file=None):
    if io_file is None:
        io_file = get_default_io()
    path = get_calibration_path(io_file)
    return list(path.glob('*.json'))


def launch_experiment(args):
    set_config('ARGS', args)
    set_config('PROFILE', args.profile)
    if args.profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()

    try:
        _main(args)
    except Exception as e:
        if args.pdb:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            log.exception(e)
            critical(None, 'Error starting experiment', str(e))

    if args.profile:
        pr.disable()
        path = get_config('LOG_ROOT') / 'main_thread.pstat'
        pr.dump_stats(path)
        stat_files = [str(p) for p in path.parent.glob('*.pstat')]
        merged_stats = pstats.Stats(*stat_files)
        merged_stats.dump_stats(path.parent / 'merged.pstat')


def get_default_io():
    '''
    Attempt to figure out the default IO configuration file

    Right now it just returns the first one found.
    '''
    available_io = list_io()
    log.debug('Found the following IO files: %r', available_io)
    if len(available_io) == 0:
        raise ValueError('No IO configured for system')
    return available_io[0]


def load_io_manifest(io_manifest=None):
    '''
    Load the IOManifest from the specified file or module

    Parameters
    ----------
    io_manifest : {str, Path, None}
        If a path (ending in `.enaml`), load the `IOManifest` from the file
        specified by the path. If a module (e.g., `psilbhb.io.badger`) load the
        `IOManifest` from the module. If None, loads the manifest returned by
        `get_default_io`.

    Returns
    -------
    io_manifest : IOManifest
        IOManifest class.
    '''
    if io_manifest is None:
        io_manifest = get_default_io()
    if str(io_manifest).endswith('.enaml'):
        klass = load_manifest_from_file(io_manifest, 'IOManifest')
    else:
        klass = load_manifest(f'{io_manifest}.IOManifest')
    return klass


def get_default_calibration(io_file):
    available_calibrations = list_calibrations(io_file)
    for calibration in available_calibrations:
        if calibration.stem == 'default':
            return calibration
    raise ValueError('No default calibration configured for system')


def load_paradigm_descriptions():
    '''
    Loads paradigm descriptions
    '''
    from psi.experiment.api import ParadigmDescription

    default = list_paradigm_descriptions()
    descriptions = get_config('PARADIGM_DESCRIPTIONS', default)
    for description in descriptions:
        importlib.import_module(description)


def add_default_options(parser):
    import argparse

    class CalibrationAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            path = Path(value)
            if not path.exists():
                path = namespace.io / path
                path = path.with_suffix('.json')
                if not path.exists():
                    raise ValueError('%s does not exist'.format(value))
            setattr(namespace, self.dest, value)

    try:
        default_io = get_default_io()
    except ValueError:
        default_io = None
    parser.add_argument('pathname', type=str, help='Filename', nargs='?')
    parser.add_argument('--io', type=str, default=default_io,
                        help='Hardware configuration')
    parser.add_argument('--calibration', type=str, help='Hardware calibration',
                        action=CalibrationAction)
    parser.add_argument('--debug', default=True, action='store_true',
                        help='Debug mode?')
    parser.add_argument('--debug-warning', default=False, action='store_true',
                        help='Show warnings?')
    parser.add_argument('--debug-level-console', type=str, default='INFO',
                        help='Logging level for console')
    parser.add_argument('--debug-level-file', type=str, default='INFO',
                        help='Logging level for file')
    parser.add_argument('--debug-exclude', type=str, nargs='*',
                        help='Names to exclude from debugging')
    parser.add_argument('--pdb', default=False, action='store_true',
                        help='Autolaunch PDB?')
    parser.add_argument('--no-preferences', default=False, action='store_true',
                        help="Don't load existing preference files")
    parser.add_argument('--no-layout', default=False, action='store_true',
                        help="Don't load existing layout files")
    parser.add_argument('-c', '--commands', nargs='+', default=[],
                        help='Commands to invoke')
    parser.add_argument('-p', '--preferences', type=str, nargs='?',
                        help='Preferences file')
    parser.add_argument('-l', '--layout', type=str, nargs='?',
                        help='Layout file')
    parser.add_argument('--profile', action='store_true', help='Profile app')


def parse_args(parser):
    args = parser.parse_args()
    if args.calibration is None:
        try:
            args.calibration = get_default_calibration(args.io)
        except ValueError as e:
            log.warn(str(e))
    return args


def list_io_templates():
    io_template_path = Path(__file__).parent.parent / 'templates' / 'io'
    return list(io_template_path.glob('*.enaml'))


def list_paradigm_descriptions():
    '''
    List default paradigms descriptions provided by psiexperiment

    Returns
    -------
    modules : list of strings
        List of strings identifying the module path for the description
    '''
    paradigm_path = Path(__file__).parent.parent / 'paradigms' / 'descriptions'
    result = []
    for filename in paradigm_path.glob('*.py'):
        s = str(filename.with_suffix(''))
        i = s.rfind('psi')
        module = s[i:].replace('/', '.').replace('\\', '.')
        result.append(module)
    return result


def config():
    import argparse
    import psi

    # Identify all the possible hardware configurations. Thoe prefixed by an
    # underscore are not available for running directly as they need to be
    # modified before use. All hardware configurations can be used as a
    # template for a skeleton that's copied to the IO_ROOT folder.
    io_template_paths = list_io_templates()
    io_skeleton_choices = [p.stem.strip('_') for p in io_template_paths]
    io_choices = [p.stem for p in io_template_paths if not p.stem.startswith('_')]

    paradigms = list_paradigm_descriptions()
    paradigm_choices = {p.rsplit('.', 1)[1]: p for p in paradigms}

    def show_config(args):
        print(psi.get_config_file())

    def create_config(args):
        base_directory = args.base_directory.rstrip('\\')
        if args.paradigm_description is None:
            paradigms = []
        else:
            paradigms = [paradigm_choices.get(p, p) \
                         for p in args.paradigm_description]

        psi.create_config(base_directory=base_directory, standard_io=args.io,
                          paradigm_descriptions=paradigms)
        if args.base_directory:
            psi.create_config_dirs()

    def create_folders(args):
        psi.create_config_dirs()

    def create_io(args):
        i = io_skeleton_choices.index(args.template)
        template = io_template_paths[i].name
        psi.create_io_manifest(template)

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
        description='Show location of config file.',
    )
    show.set_defaults(func=show_config)

    create = subparsers.add_parser('create')
    create.set_defaults(func=create_config)
    create.add_argument(
        '--base-directory',
        type=str,
        help='Root directory to store data and settings for psiexperiment.'
    )
    create.add_argument(
        '--io',
        nargs='*',
        type=str,
        help='Default hardware configurations.',
    )
    create.add_argument(
        '--paradigm-description',
        nargs='*',
        type=str,
        help=f'''Default paradigm descriptions. Can either specify a
        fully-qualified module path (e.g., psilbhb.paradigms.lbhb) or the names
        of one of the built-in paradigms. Available built-in paradigms include
        {', '.join(paradigm_choices.keys())}.'
        '''
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
        choices=io_skeleton_choices,
        help='Template to use for hardware configuration skeleton.',
    )

    args = parser.parse_args()
    args.func(args)
