import logging
log = logging.getLogger(__name__)

from atom.api import Value

import enaml
from enaml.application import deferred_call
from enaml.workbench.api import Workbench

with enaml.imports():
    from enaml.stdlib.message_box import critical
    from enaml.workbench.core.core_manifest import CoreManifest
    from enaml.workbench.ui.ui_manifest import UIManifest

    from psi.context.manifest import ContextManifest
    from psi.data.manifest import DataManifest
    from psi.experiment.manifest import ExperimentManifest
    from psi.token.manifest import TokenManifest
    from psi.controller.calibration.manifest import CalibrationManifest

    from . import error_style

from psi import set_config
from psi.core.enaml.api import load_manifest, load_manifest_from_file


class PSIWorkbench(Workbench):

    io_manifest_class = Value()

    def register_core_plugins(self, io_manifest, controller_manifests):
        # Note, the get_plugin calls appear to be necessary to properly
        # initialize parts of the application before new plugins are loaded.
        # This is likely some sort of bug or poor design on my part.
        self.register(ExperimentManifest())
        self.register(ContextManifest())
        self.register(DataManifest())
        self.register(TokenManifest())
        self.register(CalibrationManifest())
        self.register(CoreManifest())
        self.register(UIManifest())

        self.get_plugin('enaml.workbench.ui')
        self.get_plugin('enaml.workbench.core')

        self.io_manifest_class = load_manifest_from_file(io_manifest, 'IOManifest')
        io_manifest = self.io_manifest_class()
        self.register(io_manifest)

        manifests = [io_manifest]
        for manifest in controller_manifests:
            log.info('Registering %r', manifest)
            manifests.append(manifest)
            self.register(manifest)

        # Required to bootstrap plugin loading
        controller = self.get_plugin('psi.controller')
        self.get_plugin('psi.controller.calibration')
        context = self.get_plugin('psi.context')

        # Now, bind context to any manifests that want it (TODO, I should
        # have a core PSIManifest that everything inherits from so this
        # check isn't necessary).
        for manifest in manifests:
            manifest.C = context.lookup
            manifest.context = context
            manifest.controller = controller

    def start_workspace(self,
                        experiment_name,
                        base_path=None,
                        workspace='psi.experiment.ui.workspace',
                        commands=None,
                        load_preferences=True,
                        load_layout=True,
                        preferences_file=None,
                        layout_file=None,
                        calibration_file=None):

        ui = self.get_plugin('enaml.workbench.ui')
        core = self.get_plugin('enaml.workbench.core')
        commands = [] if commands is None else [(c,) for c in commands]

        # Load preferences
        if load_preferences and preferences_file is not None:
            commands.append(('psi.load_preferences', {'filename': preferences_file}))
        elif load_preferences and preferences_file is None:
            commands.append(('psi.get_default_preferences',))

        # Load layout
        if load_layout and layout_file is not None:
            commands.append(('psi.load_layout', {'filename': layout_file}))
        elif load_layout and layout_file is None:
            commands.append(('psi.get_default_layout',))

        for command in commands:
            deferred_call(core.invoke_command, *command)

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
        if base_path is None:
            ui.workspace.dock_area.style = 'error'
        ui.start_application()
