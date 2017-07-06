import logging
import importlib

from atom.api import Typed

import enaml
from enaml.application import deferred_call
from enaml.workbench.ui.api import Workspace

from psi import get_config

log = logging.getLogger(__name__)


with enaml.imports():
    from .view import ExperimentView


class ExperimentWorkspace(Workspace):

    toolbars = Typed(list, [])

    def start(self):
        log.debug('Starting experiment workspace')
        self.load_io_plugin()
        self.load_controller_plugin()
        self.load_default_plugins()
        deferred_call(self.load_toolbars)
        deferred_call(self.load_defaults)

    def load_default_plugins(self):
        with enaml.imports():
            from psi.context.manifest import ContextManifest
            from psi.context.manifest import ContextViewManifest
            from psi.data.manifest import DataManifest
            from psi.token.manifest import TokenManifest
        self.workbench.register(ContextManifest())
        self.workbench.register(DataManifest())
        self.workbench.register(TokenManifest())
        self.workbench.get_plugin('psi.controller')
        self.workbench.register(ContextViewManifest())

    def load_io_plugin(self):
        with enaml.imports():
            args = get_config('ARGS')
            module_name = 'psi.application.io.{}'.format(args.io)
            module = importlib.import_module(module_name)
            self.workbench.register(module.IOManifest())

    def load_controller_plugin(self):
        with enaml.imports():
            args = get_config('ARGS')
            module_name, class_name = args.controller.rsplit('.', 1)
            module = importlib.import_module(module_name)
            manifest = getattr(module, class_name)
            self.workbench.register(manifest())

    def load_defaults(self):
        core = self.workbench.get_plugin('enaml.workbench.core')
        args = get_config('ARGS')
        if not args.no_preferences:
            core.invoke_command('psi.get_default_preferences')
        if not args.no_layout:
            core.invoke_command('psi.get_default_layout')

    def load_toolbars(self):
        experiment = self.workbench.get_plugin('psi.experiment')
        self.toolbars = experiment._toolbars
    def _default_content(self):
        return ExperimentView()

    @property
    def dock_area(self):
        return self.content.find('dock_area')
