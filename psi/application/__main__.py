import logging.config
log = logging.getLogger(__name__)

import pdb
import traceback
import warnings
import sys
import argparse
import os.path
import datetime as dt

import enaml

from psi import get_config, set_config


experiments = {
    'appetitive_gonogo_food': 'psi.application.experiment.appetitive.ControllerManifest',
    'abr': 'psi.application.experiment.abr.ControllerManifest',
    'noise_exposure': 'psi.application.experiment.noise_exposure.ControllerManifest',
}


def configure_logging(filename=None):
    time_format = '[%(relativeCreated)d %(thread)d %(name)s - %(levelname)s] %(message)s'
    simple_format = '%(thread)d %(name)s - %(message)s'

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
            'psi': {'level': 'TRACE'},
            'daqengine': {'level': 'TRACE'},
            'psi.core.chaco': {'level': 'INFO'},
            'psi.controller.engine': {'level': 'TRACE'},
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
            'level': 'TRACE',
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
    ui.select_workspace('psi.experiment.workspace')
    ui.show_window()
    ui.start_application()


def main():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('experiment', type=str, help='Experiment to run',
                        choices=experiments.keys())
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
    args = parser.parse_args()

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


if __name__ == '__main__':
    main()
