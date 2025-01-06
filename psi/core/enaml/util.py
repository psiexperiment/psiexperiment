import logging
log = logging.getLogger(__name__)

import importlib.util
from pathlib import Path

from enaml.colors import Color
from enaml.core import import_hooks
from enaml.qt.QtGui import QColor


def load_enaml_module_from_file(path):
    '''
    Load file as an enaml module.
    '''
    path = Path(path)
    name = path.with_suffix('').name
    search_path = [str(path.parent)]
    spec = import_hooks.EnamlImporter.find_spec(name, path=search_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest_from_file(path, manifest_name):
    module = load_enaml_module_from_file(path)
    return getattr(module, manifest_name)


def load_manifests(objects, workbench):
    '''
    Recursively load manifests for all PSIConbtribution subclasses in hierarchy
    '''
    from .api import PSIContribution
    for o in objects:
        if isinstance(o, PSIContribution):
            o.load_manifest(workbench)
            load_manifests(o.children, workbench)
        elif isinstance(o, list):
            load_manifests(o, workbench)
        elif hasattr(o, 'children'):
            load_manifests(o.children, workbench)


def make_color(color, alpha=None):
    if isinstance(color, (tuple, list)):
        obj = QColor(*color)
    elif isinstance(color, str):
        obj = QColor()
        obj.setNamedColor(color)
    elif isinstance(color, Color):
        return QColor(color.red, color.green, color.blue, color.alpha)
    else:
        raise ValueError('Unknown color %r', color)
    if alpha is not None:
        obj.setAlphaF(alpha)
        if not obj.isValid():
            raise ValueError('Cannot create QColor from %r', color)
    return obj


