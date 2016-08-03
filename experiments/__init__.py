import os
os.environ['ETS_TOOLKIT'] = 'qt4'

import logging.config
import threading
import tables as tb

import enaml
from enaml.workbench.api import Workbench


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
            'neurogen': {'level': 'TRACE'},
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


def monkeypatch_pytables():
    monkeypatch = [
        tb.Table.append,
        tb.Table.read,
        tb.Table.read_where,
        tb.EArray.append,
        tb.Array.__getitem__,
    ]

    def secure_lock(f, lock):
        def wrapper(*args, **kwargs):
            with lock:
                return f(*args, **kwargs)
        return wrapper

    lock = threading.Lock()
    for im in monkeypatch:
        wrapped_im = secure_lock(im, lock)
        setattr(im.im_class, im.im_func.func_name, wrapped_im)


def initialize_default(extra_manifests,
                       workspace='psi.experiment.workspace'):

    # Important! Needed to fix some read/write concurrency issues with the
    # PyTables library.
    monkeypatch_pytables()

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
