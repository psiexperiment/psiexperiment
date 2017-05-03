import logging.config
log = logging.getLogger(__name__)

import pdb
import traceback
import warnings
import sys
import re
import argparse
import os.path
import warnings
import datetime as dt

import tables as tb

from enaml.application import deferred_call
from enaml.qt.qt_application import QtApplication

from psi import application
from psi import get_config, set_config


experiment_descriptions = {
    'passive': 'psi.controller.passive_manifest.PassiveManifest',
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


def get_base_path(dirname, experiment):
    if dirname == '<memory>':
        m = 'All data will be destroyed at end of experiment'
        log.warn(m)
        base_path = '<memory>'
    else:
        base_path = os.path.join(dirname, experiment)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        # Find out the next session from the YAML file.
        settings_root = get_config('SETTINGS_ROOT')
        config_file = os.path.join(settings_root, '.bcolz_store')

        session_name = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = os.path.join(base_path, session_name)
        os.makedirs(base_path)

    return base_path


def run(args):
    app = QtApplication()

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

    experiment_description = [experiment_descriptions[args.experiment]]
    manifests = [m() for m in application.get_manifests(experiment_description)]
    manifests.insert(0, application.get_io_manifest(args.io)())

    workbench, plugin_ids = application.initialize_workbench(manifests)
    core = workbench.get_plugin('enaml.workbench.core')
    base_path = get_base_path(args.pathname, args.experiment)
    core.invoke_command('psi.data.set_base_path', {'base_path': base_path})

    core.invoke_command('enaml.workbench.ui.select_workspace',
                        {'workspace': 'psi.experiment.workspace'})

    # the application is started (the application needs to load the plugins
    # first). For example, the controller IO extension point will automatically
    # load a series of manifests based on the equipment described in the IO
    # manifest. First, we need to load the experiment plugin to ensure that it
    # initializes everything properly. Then we can load the default layout and
    # preferences.
    log.info('Loading experiment plugin')
    ui = workbench.get_plugin('enaml.workbench.ui')
    ui.show_window()

    workbench.get_plugin('psi.controller')
    workbench.get_plugin('psi.experiment')

    # We need to use deferred call to ensure these commands are invoked *after*
    if not args.no_preferences:
        deferred_call(core.invoke_command, 'psi.get_default_preferences')
    if not args.no_layout:
        deferred_call(core.invoke_command, 'psi.get_default_layout')

    log.debug('Starting application')
    ui.start_application()

    # Unregister all plugins (ensures that stop method is called if needed)
    for plugin_id in plugin_ids[::-1]:
        workbench.unregister(plugin_id)


def main():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('experiment', type=str, help='Experiment to run',
                        choices=experiment_descriptions.keys())
    parser.add_argument('pathname', type=str, help='Filename', nargs='?',
                        default='<memory>')
    parser.add_argument('--io', type=str, default=None,
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

    try:
        run(args)
    except:
        if args.pdb:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise


if __name__ == '__main__':
    main()
