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
        deferred_call(self.load_toolbars)
        deferred_call(self.plugins_started)

    def plugins_started(self):
        controller = self.workbench.get_plugin('psi.controller')
        controller.invoke_actions('plugins_started')

    def load_toolbars(self):
        experiment = self.workbench.get_plugin('psi.experiment')
        self.toolbars = experiment._toolbars

    def _default_content(self):
        return ExperimentView()

    @property
    def dock_area(self):
        return self.content.find('dock_area')
