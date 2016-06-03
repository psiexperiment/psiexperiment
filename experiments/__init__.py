import logging.config
import warnings

import enaml
from enaml.workbench.api import Workbench


def configure_logging(filename=None):
    time_format = '[%(asctime)s] :: %(name)s - %(levelname)s - %(message)s'
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
            '__main__': {'level': 'DEBUG'},
            'neurogen': {'level': 'ERROR'},
            'psi': {'level': 'DEBUG'},
            'experiments': {'level': 'DEBUG'},
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
            'level': 'DEBUG',
        }
        logging_config['root']['handlers'].append('file')
    logging.config.dictConfig(logging_config)


def initialize_default(extra_manifests,
                       workspace='psi.experiment.workspace'):

    with enaml.imports():
        from enaml.workbench.core.core_manifest import CoreManifest
        from enaml.workbench.ui.ui_manifest import UIManifest

        from psi.context.manifest import ContextManifest
        from psi.data.manifest import DataManifest
        from psi.experiment.manifest import ExperimentManifest

    workbench = Workbench()
    workbench.register(CoreManifest())
    workbench.register(UIManifest())
    workbench.register(ContextManifest())
    workbench.register(DataManifest())
    workbench.register(ExperimentManifest())
    for manifest in extra_manifests:
        workbench.register(manifest())

    return workbench
