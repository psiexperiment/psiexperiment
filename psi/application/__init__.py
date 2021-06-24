import logging.config
log = logging.getLogger(__name__)

import datetime as dt
from glob import glob
import importlib
import os.path
from pathlib import Path
import pdb
import sys
import traceback
import warnings

import enaml
from enaml.application import deferred_call
with enaml.imports():
    from enaml.stdlib.message_box import critical

from psi import get_config, set_config

mesg_template = '''
A critical exception has occurred. While we do our best to prevent these
issues, they sometimes happen. We are now attempting to shut down the program
gracefully so acquired data can be saved. Please notify the developers.

The error message is:
{}

{}
'''



class ExceptionHandler:

    def __init__(self):
        self.workspace = None
        self.logfile = None

    def __call__(self, *args):
        log.exception("Uncaught exception", exc_info=args)
        if self.workbench is not None:
            controller = self.workbench.get_plugin('psi.controller')
            controller.stop_experiment()
            ui = self.workbench.get_plugin('enaml.workbench.ui')
            if self.logfile is not None:
                log_mesg = f'The log file has been saved to {self.logfile}'
            else:
                log_mesg = 'Unfortunately, no log file was saved.'

            mesg = mesg_template.format(args[1], log_mesg)
            deferred_call(critical, ui.window, 'Oops :(', mesg)

        sys.__excepthook__(*args)


exception_handler = ExceptionHandler()


def configure_logging(level_console=None, level_file=None, filename=None,
                      debug_exclude=None):

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

    fmt = '{levelname:10s}: {threadName:11s} - {name:40s}:: {message}'
    formatter = logging.Formatter(fmt, style='{')

    if level_console is not None:
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
    sys.excepthook = _exception_notifier


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
                              calibration_file=args.calibration)


def list_preferences(experiment):
    if not isinstance(experiment, str):
        experiment = experiment.name
    p_root = get_config('PREFERENCES_ROOT') / experiment
    p_wildcard = get_config('PREFERENCES_WILDCARD')
    p_glob = p_wildcard[:-1].split('(')[1]
    matches = p_root.glob(p_glob)
    return sorted(Path(p) for p in matches)


def list_io():
    io_path = get_config('IO_ROOT')
    return list(io_path.glob('*.enaml'))


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

    Rules
    -----
    * If no files are defined, raise ValueError
    * If only one file is defined, return it
    * If more than one file is defined, check for one named "default" first. If
      none are named "default", check to see if one matches the hostname.
    * Finally, raise ValueError if it cannot define a reasonalbe default.
    '''
    system = get_config('SYSTEM')
    available_io = list_io()
    log.debug('Found the following IO files: %r', available_io)

    if len(available_io) == 0:
        raise ValueError('No IO configured for system')
    elif len(available_io) == 1:
        return available_io[0]

    # First, check for one named "default"
    for io in available_io:
        if 'default' in io.stem.lower():
            return io

    # Next, check to see if it matches hostname
    for io in available_io:
        if io.stem == system:
            return io

    # Give up
    raise ValueError('Could not identify default IO configuration')


def get_default_calibration(io_file):
    available_calibrations = list_calibrations(io_file)
    for calibration in available_calibrations:
        if calibration.stem == 'default':
            return calibration
    raise ValueError('No default calibration configured for system')


def add_default_options(parser):
    import argparse

    class IOAction(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            path = Path(value)
            if not path.exists():
                path = get_config('IO_ROOT') / value
                path = path.with_suffix('.enaml')
                if not path.exists():
                    raise ValueError('{} does not exist'.format(value))
            setattr(namespace, self.dest, path)

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
                        help='Hardware configuration', action=IOAction)
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
    parser.add_argument('--profile', action='store_true', help='Profile app')


def parse_args(parser):
    args = parser.parse_args()
    print(args)
    if args.calibration is None:
        try:
            args.calibration = get_default_calibration(args.io)
        except ValueError as e:
            log.warn(str(e))
    return args


def config():
    import argparse
    import psi

    def show_config(args):
        print(psi.get_config_file())

    def create_config(args):
        psi.create_config(base_directory=args.base_directory)
        if args.base_directory:
            psi.create_config_dirs()

    def create_folders(args):
        psi.create_config_dirs()

    def create_io(args):
        psi.create_io_manifest(args.template)

    parser = argparse.ArgumentParser('psi-config')
    subparsers = parser.add_subparsers(dest='cmd')
    subparsers.required = True

    show = subparsers.add_parser('show')
    show.set_defaults(func=show_config)

    create = subparsers.add_parser('create')
    create.set_defaults(func=create_config)
    create.add_argument('--base-directory', type=str)

    make = subparsers.add_parser('create-folders')
    make.set_defaults(func=create_folders)

    io_template_path = Path(__file__).parent.parent / 'templates' / 'io'
    io_choices = [p.stem for p in io_template_path.glob('*.enaml')]
    io = subparsers.add_parser('create-io')
    io.set_defaults(func=create_io)
    io.add_argument('template', type=str, choices=io_choices)

    args = parser.parse_args()
    args.func(args)
