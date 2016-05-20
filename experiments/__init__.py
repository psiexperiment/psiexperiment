import warnings

import enaml
from enaml.workbench.api import Workbench


def initialize_default(extra_manifests,
                       workspace='psi.experiment.workspace'):

    with enaml.imports():
        from enaml.workbench.core.core_manifest import CoreManifest
        from enaml.workbench.ui.ui_manifest import UIManifest

        from psi.context.manifest import ContextManifest
        from psi.data.manifest import DataManifest
        from psi.experiment.manifest import ExperimentManifest

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        workbench = Workbench()
        workbench.register(CoreManifest())
        workbench.register(UIManifest())
        workbench.register(ContextManifest())
        workbench.register(DataManifest())
        workbench.register(ExperimentManifest())
        for manifest in extra_manifests:
            workbench.register(manifest())

        core = workbench.get_plugin('enaml.workbench.core')
        ui = workbench.get_plugin('enaml.workbench.ui')
        ui.show_window()
        core.invoke_command('enaml.workbench.ui.select_workspace',
                            {'workspace': workspace})

        experiment = workbench.get_plugin('psi.experiment')
        return workbench
