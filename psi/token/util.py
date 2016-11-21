import os.path

import enaml


def find_manifests(directory):
    modules = []
    for root, _, filenames in os.walk(directory):
        rroot = os.path.relpath(root, directory)
        if not rroot.strip('.'):
            continue
        for filename in filenames:
            if filename.endswith('manifest.enaml'):
                filename = os.path.join('psi', 'token', rroot, filename[:-6])
                modules.append(filename.replace(os.path.sep, '.'))
    return modules


def autoload_tokens():
    directory = os.path.dirname(__file__)
    manifests = []
    with enaml.imports():
        for manifest in find_manifests(directory):
            module = importlib.import_module(manifest)
            for child in dir(module):
                if child.endswith('Manifest') and child != 'PluginManifest':
                    manifests.append(getattr(module, child))
    return manifests
