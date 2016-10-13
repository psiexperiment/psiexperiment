import importlib
import os.path

import enaml

from psi.core.enaml.manifest import PSIManifest


def _filter_manifests(manifests, directory, files):
    for f in files:
        if f.endswith('manifest.enaml'):
            manifests.append(os.path.join(directory, f))


def _to_module(manifest, base_path):
    relpath = os.path.relpath(manifest, base_path)
    dotpath = relpath.replace(os.path.sep, '.')
    return os.path.splitext(dotpath)[0]


def _list_manifests(base_path):
    manifests = []
    os.path.walk(base_path, _filter_manifests, manifests)
    return [_to_module(m, base_path) for m in manifests]


def _load_manifests(manifests):
    with enaml.imports():
        modules = [importlib.import_module(m) for m in manifests]
        return modules


def load_manifests(base_path):
    names = _list_manifests(base_path)
    return _load_manifests(names)


def _find_subclasses(cls):
    subclasses = cls.__subclasses__() 
    nested_subclasses = [ns for s in subclasses for ns in _find_subclasses(s)]
    return subclasses + nested_subclasses


def _find_bases(cls):
    if cls is PSIManifest:
        return []
    bases = list(cls.__bases__)
    nested_bases = [nb for b in bases for nb in _find_bases(b)]
    return bases + nested_bases


def find_extensions(controller_manifest):
    base_classes = _find_bases(controller_manifest)
    subclasses = _find_subclasses(PSIManifest)
    return [s for s in subclasses if controller in s().supplements]


def find_manifests(pattern):
    subclasses = _find_subclasses(PSIManifest)
    for subclass in subclasses:
        print subclass().id


with enaml.imports():
    from experiments.appetitive.manifest import ControllerManifest

base_path = 'c:/users/bburan/projects/dev/psiexperiment'
#controller = 'Appetitive Go-Nogo'
# This will find all manifests that we can load. This is required before we can
# do anything else!
load_manifests(base_path)

print _find_bases(ControllerManifest)

# Find all extensions compatible with the current controller.
#print find_extensions(controller)
