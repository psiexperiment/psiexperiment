import logging.config
log = logging.getLogger(__name__)

import pdb
import traceback
import warnings
import sys
import os.path
import datetime as dt
from glob import glob

import enaml
from enaml.application import deferred_call

from psi import get_config, set_config


experiments = {
    'appetitive_gonogo_food': 'psi.application.experiment.appetitive.ControllerManifest',
    'abr': 'psi.application.experiment.abr_with_temperature.ControllerManifest',
    'abr-debug': 'psi.application.experiment.abr_debug.ControllerManifest',
    'noise_exposure': 'psi.application.experiment.noise_exposure.ControllerManifest',
    'efr': 'psi.application.experiment.efr.ControllerManifest',
    'dual_efr': 'psi.application.experiment.dual_efr.ControllerManifest',
    'calibration': 'psi.application.experiment.calibration.ControllerManifest',
    'pt_calibration': 'psi.application.experiment.pt_calibration.ControllerManifest',
    'pt_calibration_chirp': 'psi.application.experiment.pt_calibration_chirp.ControllerManifest',
    'pistonphone_calibration': 'psi.application.experiment.pistonphone_calibration.ControllerManifest',
    'dpoae': 'psi.application.experiment.dpoae.ControllerManifest',
}


def configure_logging(filename=None):
    time_format = '[%(relativeCreated)d %(thread)d %(name)s - %(levelname)s] %(message)s'
    simple_format = '%(relativeCreated)8d %(thread)d %(name)s - %(message)s'

    logging_config = {
        'version': 1,
        'formatters': {
            'time': {'format': time_format},
            'simple': {'format': simple_format},
            },
        'handlers': {
            # This is what gets printed out to the console
            'console': {
                'class': 'psi.core.logging.colorstreamhandler.ColorStreamHandler',
                'formatter': 'simple',
                'level': 'TRACE',
                },
            },
        'loggers': {
            '__main__': {'level': 'INFO'},
            'psi': {'level': 'INFO'},
            },
        'root': {
            'handlers': ['console'],
            },
        }

    if filename is not None:
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'formatter': 'time',
            'filename': filename,
            'mode': 'w',
            'encoding': 'UTF-8',
            'level': 'DEBUG',
        }
        logging_config['root']['handlers'].append('file')
    logging.config.dictConfig(logging_config)


def warn_with_traceback(message, category, filename, lineno, file=None,
                        line=None):
    traceback.print_stack()
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
        configure_logging(os.path.join(log_root, filename))
        log.debug('Logging configured')
        log.info('Logging information captured in {}'.format(filename))
        if args.debug_warning:
            warnings.showwarning = warn_with_traceback

    from enaml.workbench.api import Workbench
    with enaml.imports():
        from enaml.workbench.core.core_manifest import CoreManifest
        from enaml.workbench.ui.ui_manifest import UIManifest
        from psi.experiment.manifest import ExperimentManifest

    workbench = Workbench()
    workbench.register(ExperimentManifest())
    workbench.register(CoreManifest())
    workbench.register(UIManifest())

    ui = workbench.get_plugin('enaml.workbench.ui')
    core = workbench.get_plugin('enaml.workbench.core')
    ui.select_workspace('psi.experiment.workspace')

    if args.pathname is None:
        log.warn('All data will be destroyed at end of experiment')

    ui.show_window()

    if args.preferences is not None:
        deferred_call(core.invoke_command, 'psi.load_preferences',
                    {'filename': args.preferences})

    for command in args.commands:
        deferred_call(core.invoke_command, command)
    ui.start_application()


def list_preferences(experiment):
    p_root = get_config('PREFERENCES_ROOT')
    p_wildcard = get_config('PREFERENCES_WILDCARD')
    p_glob = p_wildcard[:-1].split('(')[1]
    p_search = os.path.join(p_root, experiment, p_glob)
    return glob(p_search)


def list_io():
    hostname = get_config('SYSTEM')
    from . import io
    from glob import iglob
    base_path = os.path.dirname(io.__file__)
    search_path = os.path.join(base_path, '*{}*'.format(hostname))
    return [os.path.basename(f)[:-6] for f in iglob(search_path)]


def launch_experiment(args):
    # Map to the actual controller module.
    args.controller = experiments[args.experiment]
    set_config('ARGS', args)
    try:
        _main(args)
    except:
        if args.pdb:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise


def add_default_options(parser):
    parser.add_argument('pathname', type=str, help='Filename', nargs='?')
    parser.add_argument('--io', type=str, default=get_config('SYSTEM'),
                        help='Hardware configuration')
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
