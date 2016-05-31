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

        #from enaml.qt.QtGui import QApplication, QFont
        #ui = workbench.get_plugin('enaml.workbench.ui')
        #app = QApplication.instance()
        #app.setFont(QFont('DejaVu Sans Mono'))

        return workbench
