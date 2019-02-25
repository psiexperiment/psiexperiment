import logging
log = logging.getLogger(__name__)

import enaml
from enaml.application import deferred_call
from enaml.workbench.api import Workbench

with enaml.imports():
    from enaml.stdlib.message_box import critical

from psi import set_config
from psi.application import load_manifest, load_manifest_from_file


class PSIWorkbench(Workbench):

    def register_core_plugins(self, io_manifest, controller_manifests):
        # Note, the get_plugin calls appear to be necessary to properly
        # initialize parts of the application before new plugins are loaded.
        # This is likely some sort of bug or poor design on my part.
        with enaml.imports():
            from enaml.workbench.core.core_manifest import CoreManifest
            from enaml.workbench.ui.ui_manifest import UIManifest
            from psi.experiment.manifest import ExperimentManifest

            self.register(ExperimentManifest())
            self.register(CoreManifest())
            self.register(UIManifest())
            self.get_plugin('enaml.workbench.ui')
            self.get_plugin('enaml.workbench.core')

            manifest_class = load_manifest_from_file(io_manifest, 'IOManifest')
            self.register(manifest_class())

            for manifest in controller_manifests:
                manifest_class = load_manifest(manifest)
                self.register(manifest_class())

            from psi.context.manifest import ContextManifest
            from psi.data.manifest import DataManifest
            from psi.token.manifest import TokenManifest

            self.register(ContextManifest())
            self.register(DataManifest())
            self.register(TokenManifest())

            self.get_plugin('psi.controller')

            from psi.context.manifest import ContextViewManifest
            self.register(ContextViewManifest())

    def start_workspace(self,
                        experiment_name,
                        base_path=None,
                        workspace='psi.experiment.workspace',
                        commands=None,
                        load_preferences=True,
                        load_layout=True,
                        preferences_file=None,
                        layout_file=None,
                        calibration_file=None):

        # TODO: Hack alert ... don't store this information in a shared config
        # file. It's essentially a global variable.
        set_config('EXPERIMENT', experiment_name)

        ui = self.get_plugin('enaml.workbench.ui')
        core = self.get_plugin('enaml.workbench.core')

        # Load preferences
        if load_preferences and preferences_file is not None:
            deferred_call(core.invoke_command, 'psi.load_preferences',
                          {'filename': preferences_file})
        elif load_preferences and preferences_file is None:
            deferred_call(core.invoke_command, 'psi.get_default_preferences')

        # Load layout
        if load_layout and layout_file is not None:
            deferred_call(core.invoke_command, 'psi.load_layout',
                          {'filename': layout_file})
        elif load_layout and layout_file is None:
            deferred_call(core.invoke_command, 'psi.get_default_layout')

        # Exec commands
        if commands is not None:
            for command in commands:
                deferred_call(core.invoke_command, command)

        controller = self.get_plugin('psi.controller')
        if base_path is not None:
            controller.register_action('experiment_prepare',
                                       'psi.data.set_base_path',
                                       {'base_path': base_path})

        if calibration_file is not None:
            controller.load_calibration(calibration_file)

        # Now, open workspace
        ui.select_workspace(workspace)
        ui.show_window()
        ui.start_application()
