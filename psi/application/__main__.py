import logging.config
log = logging.getLogger(__name__)

import argparse
import os.path
import warnings

import tables as tb

experiment_descriptions = {
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
    }
}


def configure_logging(filename=None):
    time_format = '[%(relativeCreated)d %(thread)d %(name)s - %(levelname)s] %(message)s'
    simple_format = '%(name)s - %(message)s'

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
            '__main__': {'level': 'TRACE'},
            'neurogen': {'level': 'ERROR'},
            'psi': {'level': 'TRACE'},
            'experiments': {'level': 'TRACE'},
            'daqengine': {'level': 'TRACE'},
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('experiment', type=str, help='Experiment to run')
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

    with tb.open_file('c:/users/bburan/desktop/test.h5', 'w') as fh:
        core.invoke_command('psi.data.hdf_store.set_node', {'node': fh.root})

        ui = workbench.get_plugin('enaml.workbench.ui')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ui.show_window()

        ui.start_application()
