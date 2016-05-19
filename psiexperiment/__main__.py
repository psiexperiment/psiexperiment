import warnings

import enaml
from enaml.workbench.api import Workbench


if __name__ == '__main__':
    with enaml.imports():
        from enaml.workbench.core.core_manifest import CoreManifest
        from enaml.workbench.ui.ui_manifest import UIManifest

        from psiexperiment.setting.manifest import SettingManifest
        from psiexperiment.data.manifest import DataManifest
        from psiexperiment.controller.manifest import ControllerManifest
        from psiexperiment.experiment.manifest import ExperimentManifest

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        workbench = Workbench()
        workbench.register(CoreManifest())
        workbench.register(UIManifest())

        workbench.register(SettingManifest())
        workbench.register(DataManifest())
        workbench.register(ControllerManifest())
        workbench.register(ExperimentManifest())

        core = workbench.get_plugin('enaml.workbench.core')
        ui = workbench.get_plugin('enaml.workbench.ui')

        parameters = core.invoke_command('psiexperiment.setting.get_parameters')
        parameters[0].rove = True
        name = parameters[0].name

        setting = workbench.get_plugin('psiexperiment.setting')
        setting.selectors['go'].add_setting({name: 12.0})

        ui.show_window()
        core.invoke_command('enaml.workbench.ui.select_workspace',
                            {'workspace': 'psiexperiment.experiment.view'})
        core.invoke_command('psiexperiment.controller.start')
        ui.start_application()
