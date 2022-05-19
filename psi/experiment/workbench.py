import logging
log = logging.getLogger(__name__)

import tempfile

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
from psi.core.enaml import manifest


class PSIWorkbench(Workbench):

    io_manifest_class = Value()
    context_plugin = Value()
    controller_plugin = Value()

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

        if io_manifest is not None:
            self.io_manifest_class = load_manifest_from_file(io_manifest, 'IOManifest')
            io_manifest = self.io_manifest_class()
            self.register(io_manifest)
            manifests = [io_manifest]
        else:
            manifests = []

        for manifest in controller_manifests:
            log.info('Registering %r', manifest)
            manifests.append(manifest)
            self.register(manifest)

        # Required to bootstrap plugin loading
        self.controller_plugin = self.get_plugin('psi.controller')
        self.get_plugin('psi.controller.calibration')
        self.context_plugin = self.get_plugin('psi.context')

        for manifest in self._manifests.values():
            if hasattr(manifest, 'C'):
                # Now, bind information to the manifests
                manifest.C = self.context_plugin.lookup
                manifest.context = self.context_plugin
                manifest.controller = self.controller_plugin

    def register(self, manifest):
        if isinstance(manifest, str):
            manifest = load_manifest(manifest)()

        if self.context_plugin is not None and hasattr(manifest, 'C'):
            manifest.C = self.context_plugin.lookup
            manifest.context = self.context_plugin
            manifest.controller = self.controller_plugin
        super().register(manifest)

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
        controller = self.get_plugin('psi.controller')

        ui.select_workspace(workspace)
        ui.show_window()

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


        # Configure the path where the data is saved. If `is_temp` is True, the
        # path will automatically be deleted at the end of the experiment.
        if base_path is not None:
            params = {'base_path': base_path, 'is_temp': False}
        else:
            tmp_path = tempfile.mkdtemp()
            params = {'base_path': tmp_path, 'is_temp': True}
        controller.register_action('experiment_prepare',
                                   'psi.data.set_base_path', params)

        if calibration_file is not None:
            controller.load_calibration(calibration_file)

        # Now, open workspace
        if base_path is None:
            ui.workspace.dock_area.style = 'error'
        ui.start_application()
