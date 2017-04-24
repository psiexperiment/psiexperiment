import importlib

import enaml
from enaml.workbench.api import Workbench


def import_manifest(module_name):
    with enaml.imports():
        return importlib.import_module(module_name)


def get_io_manifest(system=None):
    if system is None:
        from psi import get_config
        system = get_config('SYSTEM')
    module_name = 'psi.application.io.{}'.format(system)
    module = import_manifest(module_name)
    return module.IOManifest


def get_controller_manifest(experiment):
    module_name = 'psi.application.experiment.{}'.format(experiment)
    module = import_manifest(module_name)
    return module.ControllerManifest


def get_manifests(manifest_names):
    manifests = []
    with enaml.imports():
        for manifest_name in manifest_names:
            module_name, class_name = manifest_name.rsplit('.', 1)
            module = import_manifest(module_name)
            manifests.append(getattr(module, class_name))
    return manifests


def initialize_workbench(extra_manifests,
                         workspace='psi.experiment.workspace'):

    with enaml.imports():
        from enaml.workbench.core.core_manifest import CoreManifest
        from enaml.workbench.ui.ui_manifest import UIManifest

        from psi.context.manifest import ContextManifest
        from psi.context.manifest import ContextViewManifest
        from psi.data.manifest import DataManifest
        from psi.experiment.manifest import ExperimentManifest
        from psi.token.manifest import TokenManifest

    plugin_ids = []
    def track_plugins(event):
        plugin_ids.append(event['value'])

    workbench = Workbench()
    workbench.observe('plugin_added', track_plugins)

    workbench.register(CoreManifest())
    workbench.register(UIManifest())
    workbench.register(ContextManifest())
    workbench.register(ContextViewManifest())
    workbench.register(DataManifest())
    workbench.register(ExperimentManifest())
    workbench.register(TokenManifest())

    for manifest in extra_manifests:
        workbench.register(manifest)

    return workbench, plugin_ids
