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
with enaml.imports():
    from enaml.stdlib.message_box import critical

from psi import get_config, set_config


def _exception_notifier(*args):
    log.error("Uncaught exception", exc_info=args)
    sys.__excepthook__(*args)


def configure_logging(level, filename=None):
    log = logging.getLogger()
    log.setLevel(level)
    stream_handler = logging.StreamHandler()
    log.addHandler(stream_handler)
    if filename is not None:
        file_handler = logging.FileHandler(filename, 'w', 'UTF-8')
        log.addHandler(file_handler)

    sys.excepthook = _exception_notifier


def warn_with_traceback(message, category, filename, lineno, file=None,
                        line=None):
    #traceback.print_stack()
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
        configure_logging('DEBUG', os.path.join(log_root, filename))

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

    workbench.start_workspace(args.experiment,
                              args.pathname,
                              commands=args.commands,
                              load_preferences=not args.no_preferences,
                              load_layout=not args.no_layout,
                              preferences_file=args.preferences,
                              calibration_file=args.calibration)


def list_preferences(experiment):
    p_root = get_config('PREFERENCES_ROOT') / experiment.name
    p_wildcard = get_config('PREFERENCES_WILDCARD')
    p_glob = p_wildcard[:-1].split('(')[1]
    matches = p_root.glob(p_glob)
    return sorted(Path(p) for p in matches)


def list_io():
    io_path = get_config('IO_ROOT')
    return list(io_path.glob('*.enaml'))


def list_calibrations(io_file):
    io_file = Path(io_file)
    calibration_path = io_file.parent / io_file.stem
    return list(calibration_path.glob('*.json'))


def launch_experiment(args):
    set_config('ARGS', args)
    if args.profile:
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()

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
        profiler.stop()
        html = profiler.output_html()
        with open('profile.html', 'w') as fh:
            fh.write(html)


def get_default_io():
    system = get_config('SYSTEM')
    available_io = list_io()
    for io in available_io:
        if io.stem == system:
            return io
    return None
    #raise ValueError('No IO configured for system')


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
                    raise ValueError('%s does not exist'.format(value))
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

    parser.add_argument('pathname', type=str, help='Filename', nargs='?')
    parser.add_argument('--io', type=str, default=get_default_io(),
                        help='Hardware configuration', action=IOAction)
    parser.add_argument('--calibration', type=str, help='Hardware calibration',
                        action=CalibrationAction)
    parser.add_argument('--debug', default=True, action='store_true',
                        help='Debug mode?')
    parser.add_argument('--debug-warning', default=False, action='store_true',
                        help='Debug mode?')
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
        psi.create_io_manifest()

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

    io = subparsers.add_parser('create-io')
    io.set_defaults(func=create_io)

    args = parser.parse_args()
    args.func(args)
