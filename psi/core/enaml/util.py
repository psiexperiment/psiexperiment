import importlib.util

from pathlib import Path

import enaml
from enaml.core import import_hooks


def load_manifest(manifest_path):
    try:
        module_name, manifest_name = manifest_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, manifest_name)
    except AttributeError as e:
        raise ImportError() from e


def load_enaml_module_from_file(path):
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
