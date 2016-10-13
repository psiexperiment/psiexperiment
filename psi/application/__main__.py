import argparse
import logging.config
import warnings

import tables as tb

from psi import application

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
                'level': 'DEBUG',
                },
            },
        'loggers': {
            '__main__': {'level': 'TRACE'},
            'neurogen': {'level': 'ERROR'},
            'psi': {'level': 'TRACE'},
            'experiments': {'level': 'TRACE'},
            'daqengine': {'level': 'ERROR'},
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
    args = parser.parse_args()

    experiment_description = experiment_descriptions[args.experiment]
    manifests = application.get_manifests(experiment_description['manifests'])
    manifests += [application.get_io_manifest()]
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

        filename = 'c:/users/bburan/desktop/appetitive_log.txt' 
        configure_logging(filename)
        ui.start_application()
