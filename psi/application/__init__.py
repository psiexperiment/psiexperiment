import logging.config
log = logging.getLogger(__name__)

import pdb
import traceback
import warnings
import sys
import os.path
import datetime as dt

import enaml
from enaml.application import deferred_call

from psi import get_config, set_config


experiments = {
    'appetitive_gonogo_food': 'psi.application.experiment.appetitive.ControllerManifest',
    'abr': 'psi.application.experiment.abr.ControllerManifest',
    'noise_exposure': 'psi.application.experiment.noise_exposure.ControllerManifest',
    'efr': 'psi.application.experiment.efr.ControllerManifest',
    'dual_efr': 'psi.application.experiment.dual_efr.ControllerManifest',
    'calibration': 'psi.application.experiment.calibration.ControllerManifest',
    'dpoae': 'psi.application.experiment.dpoae.ControllerManifest',
}


def configure_logging(filename=None):
    time_format = '[%(relativeCreated)d %(thread)d %(name)s - %(levelname)s] %(message)s'
    simple_format = '%(asctime)s %(thread)d %(name)s - %(message)s'

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
                'level': 'DEBUG',
                },
            },
        'loggers': {
            '__main__': {'level': 'DEBUG'},
            'psi': {'level': 'DEBUG'},
            'psi.core.chaco': {'level': 'INFO'},
            'psi.controller.engine': {'level': 'DEBUG'},
            'psi.controller.engines.nidaq': {'level': 'DEBUG'},
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


def get_base_path(dirname, experiment):
    if dirname == '<memory>':
        m = 'All data will be destroyed at end of experiment'
        log.warn(m)
        base_path = '<memory>'
    elif dirname.endswith('*'):
        base_path = os.path.join(dirname, experiment)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        settings_root = get_config('SETTINGS_ROOT')
        config_file = os.path.join(settings_root, '.bcolz_store')
        session_name = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = os.path.join(base_path, session_name)
        os.makedirs(base_path)
    else:
        base_path = dirname
        os.makedirs(base_path)
    return base_path


def _main(args):
    for config in ['LAYOUT_ROOT', 'PREFERENCES_ROOT']:
        path = get_config(config)
        new_path = os.path.join(path, args.experiment)
        set_config(config, new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

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
    else:
        # This suppresses a FutureWarning in the Chaco library that we don't
        # really need to deal with at the moment.
        warnings.simplefilter(action="ignore", category=FutureWarning)

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

    # The base path must get set before the window is shown otherwise it will
    # be too late to configure the store path for the files.
    base_path = get_base_path(args.pathname, args.experiment)
    core.invoke_command('psi.data.set_base_path', {'base_path': base_path})

    ui.show_window()

    if args.preferences is not None:
        deferred_call(core.invoke_command, 'psi.load_preferences', 
                    {'filename': args.preferences})

    for command in args.commands:
        deferred_call(core.invoke_command, command)
    ui.start_application()


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
    parser.add_argument('pathname', type=str, help='Filename', nargs='?',
                        default='<memory>')
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
