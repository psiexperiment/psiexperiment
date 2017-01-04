import logging.config
log = logging.getLogger(__name__)

import faulthandler
faulthandler.enable()

import re
import argparse
import os.path
import warnings
import datetime as dt

import tables as tb

from enaml.application import deferred_call

from psi import application
from psi import get_config, set_config


data_store_manifest = 'psi.data.bcolz_store.manifest.BColzStoreManifest'


experiment_descriptions = {
    'test': {
        'manifests': [
            #'psi.application.experiment.test.ControllerManifest',
            'psi.controller.passive_manifest.PassiveManifest',
            'psi.data.trial_log.manifest.TrialLogManifest',
            'psi.data.event_log.manifest.EventLogManifest',
            data_store_manifest,
        ]
    },
    'appetitive_gonogo_food': {
        'manifests': [
            'psi.application.experiment.appetitive.ControllerManifest',
            'psi.controller.actions.pellet_dispenser.manifest.PelletDispenserManifest',
            'psi.controller.actions.pellet_dispenser.manifest.AppetitivePelletDispenserActions',
            'psi.controller.actions.room_light.manifest.RoomLightManifest',
            'psi.controller.actions.room_light.manifest.AppetitiveRoomLightActions',
            'psi.data.trial_log.manifest.TrialLogManifest',
            'psi.data.event_log.manifest.EventLogManifest',
            'psi.data.sdt_analysis.manifest.SDTAnalysisManifest',
            data_store_manifest,
        ],
    },
    'appetitive_gonogo_water': {
        'manifests': [
            'psi.application.experiment.appetitive.ControllerManifest',
            'psi.controller.actions.NE1000.manifest.NE1000Manifest',
            'psi.controller.actions.NE1000.manifest.AppetitiveNE1000Actions',
            'psi.controller.actions.room_light.manifest.RoomLightManifest',
            'psi.controller.actions.room_light.manifest.AppetitiveRoomLightActions',
            'psi.data.trial_log.manifest.TrialLogManifest',
            'psi.data.event_log.manifest.EventLogManifest',
            'psi.data.sdt_analysis.manifest.SDTAnalysisManifest',
            data_store_manifest,
        ],
    },

    'abr': {
        'manifests': [
            'psi.application.experiment.abr.ControllerManifest',
            'psi.data.trial_log.manifest.TrialLogManifest',
            'psi.data.event_log.manifest.EventLogManifest',
            data_store_manifest,
        ],
    },

    'noise_exposure': {
        'manifests': [
            'psi.application.experiment.noise_exposure.ControllerManifest',
            'psi.data.event_log.manifest.EventLogManifest',
            data_store_manifest,
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
                'class': 'psi.core.logging.colorstreamhandler.ColorStreamHandler',
                'formatter': 'simple',
                'level': 'TRACE',
                },
            },
        'loggers': {
            '__main__': {'level': 'DEBUG'},
            'neurogen': {'level': 'ERROR'},
            'psi': {'level': 'DEBUG'},
            'experiments': {'level': 'DEBUG'},
            'daqengine': {'level': 'INFO'},
            'psi.core.chaco': {'level': 'INFO'},
            'psi.controller.engine': {'level': 'INFO'},
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
    parser.add_argument('pathname', type=str, help='Filename', nargs='?',
                        default='<memory>')
    parser.add_argument('--io', type=str, default=None,
                        help='Hardware configuration')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug mode?')
    args = parser.parse_args()

    for config in ['LAYOUT_ROOT', 'PREFERENCES_ROOT']:
        path = get_config(config)
        new_path = os.path.join(path, args.experiment)
        set_config(config, new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    if args.debug:
        dt_string = dt.datetime.now().strftime('%Y-%m-%d %H%M') 
        filename = '{} {}'.format(dt_string, args.experiment)
        log_root = get_config('LOG_ROOT')
        configure_logging(os.path.join(log_root, filename))
        log.debug('Logging configured')

    experiment_description = experiment_descriptions[args.experiment]
    manifests = application.get_manifests(experiment_description['manifests'])
    manifests += [application.get_io_manifest(args.io)]
    workbench = application.initialize_workbench(manifests)

    core = workbench.get_plugin('enaml.workbench.core')
    core.invoke_command('enaml.workbench.ui.select_workspace',
                        {'workspace': 'psi.experiment.workspace'})

    cmd = 'psi.data.bcolz_store.prepare_filesystem'
    parameters = {'pathname': args.pathname, 'experiment': args.experiment}
    core.invoke_command(cmd, parameters)
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
