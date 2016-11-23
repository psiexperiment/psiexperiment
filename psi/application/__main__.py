import logging.config
log = logging.getLogger(__name__)

import re
import argparse
import os.path
import warnings

import tables as tb

from enaml.application import deferred_call


experiment_descriptions = {
    'test': {
        'manifests': [
            'psi.application.experiment.test.ControllerManifest',
            'psi.data.hdf_store.manifest.HDFStoreManifest',
        ]
    },
    'appetitive_gonogo_food': {
        'manifests': [
            'psi.application.experiment.appetitive.ControllerManifest',
            'psi.controller.actions.pellet_dispenser.manifest.PelletDispenserManifest',
            'psi.controller.actions.room_light.manifest.RoomLightManifest',
            'psi.data.trial_log.manifest.TrialLogManifest',
            'psi.data.event_log.manifest.EventLogManifest',
            'psi.data.sdt_analysis.manifest.SDTAnalysisManifest',
            'psi.data.hdf_store.manifest.HDFStoreManifest',
        ],
    },

    'abr': {
        'manifests': [
            'psi.application.experiment.abr.ControllerManifest',
            'psi.data.trial_log.manifest.TrialLogManifest',
            'psi.data.event_log.manifest.EventLogManifest',
            'psi.data.hdf_store.manifest.HDFStoreManifest',
        ],
    },

    'noise_exposure': {
        'manifests': [
            'psi.application.experiment.noise_exposure.ControllerManifest',
            'psi.data.event_log.manifest.EventLogManifest',
            'psi.data.hdf_store.manifest.HDFStoreManifest',
        ],
    }
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
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'level': 'TRACE',
                },
            },
        'loggers': {
            '__main__': {'level': 'DEBUG'},
            'neurogen': {'level': 'ERROR'},
            'psi': {'level': 'DEBUG'},
            'psi.core.chaco': {'level': 'DEBUG'},
            'experiments': {'level': 'DEBUG'},
            'psi.controller.engine': {'level': 'TRACE'},
            'daqengine': {'level': 'DEBUG'},
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


def main():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('experiment', type=str, help='Experiment to run',
                        choices=experiment_descriptions.keys())
    parser.add_argument('filename', type=str, help='Filename', nargs='?',
                        default='<memory>')
    parser.add_argument('--io', type=str, default=None,
                        help='Hardware configuration')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug mode?')
    args = parser.parse_args()

    if args.debug:
        configure_logging()
        log.debug('Logging configured')

    from psi import application
    from psi import get_config, set_config

    for config in ['LAYOUT_ROOT', 'PREFERENCES_ROOT', 'CONTEXT_ROOT']:
        path = get_config(config)
        new_path = os.path.join(path, args.experiment)
        set_config(config, new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    experiment_description = experiment_descriptions[args.experiment]
    manifests = application.get_manifests(experiment_description['manifests'])
    manifests += [application.get_io_manifest(args.io)]
    workbench = application.initialize_workbench(manifests)

    core = workbench.get_plugin('enaml.workbench.core')
    core.invoke_command('enaml.workbench.ui.select_workspace',
                        {'workspace': 'psi.experiment.workspace'})


    cmd = 'psi.data.hdf_store.prepare_file'
    parameters = {'filename': args.filename, 'experiment': args.experiment}
    with core.invoke_command(cmd, parameters) as fh:
        ui = workbench.get_plugin('enaml.workbench.ui')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ui.show_window()

        # We need to use deferred call to ensure these commands are invoked
        # *after* the application is started (the application needs to load the
        # plugins first).
        deferred_call(core.invoke_command, 'psi.get_default_preferences')
        deferred_call(core.invoke_command, 'psi.get_default_layout')
        ui.start_application()


if __name__ == '__main__':
    main()
