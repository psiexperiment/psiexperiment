import importlib
from pathlib import Path

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
    file_info = import_hooks.make_file_info(str(path))
    importer = import_hooks.EnamlImporter(file_info)
    return importer.load_module(name)


def load_manifest_from_file(path, manifest_name):
    module = load_enaml_module_from_file(path)
    return getattr(module, manifest_name)
