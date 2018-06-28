import importlib

import enaml
from enaml.application import deferred_call
from enaml.workbench.api import Workbench

from psi import set_config


def load_manifest(manifest_path):
    module_name, manifest_name = manifest_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, manifest_name)


class PSIWorkbench(Workbench):

    def register_core_plugins(self, io_manifest, controller_manifest):
        # Note, the get_plugin calls appear to be necessary to properly
        # initialize parts of the application before new plugins are loaded.
        # This is likely some sort of bug or poor design on my part.

        with enaml.imports():
            from enaml.workbench.core.core_manifest import CoreManifest
            from enaml.workbench.ui.ui_manifest import UIManifest
            from psi.experiment.manifest import ExperimentManifest
            from psi.context.manifest import ContextManifest
            from psi.context.manifest import ContextViewManifest
            from psi.data.manifest import DataManifest
            from psi.token.manifest import TokenManifest
            IOManifest = load_manifest(io_manifest)
            ControllerManifest = load_manifest(controller_manifest)

        self.register(ExperimentManifest())
        self.register(CoreManifest())
        self.register(UIManifest())

        self.get_plugin('enaml.workbench.ui')
        self.get_plugin('enaml.workbench.core')

        self.register(IOManifest())
        self.register(ControllerManifest())

        self.register(ContextManifest())
        self.register(DataManifest())
        self.register(TokenManifest())

        self.get_plugin('psi.controller')

        self.register(ContextViewManifest())

    def start_workspace(self, experiment_name,
                        workspace='psi.experiment.workspace',
                        commands=None,
                        load_preferences=True, load_layout=True,
                        preferences_file=None, layout_file=None):

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

        # Now, open workspace
        ui.select_workspace(workspace)
        ui.show_window()
        ui.start_application()
