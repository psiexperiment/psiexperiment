import logging
log = logging.getLogger(__name__)

from atom.api import Typed

from enaml.application import deferred_call
from enaml.widgets.api import Container, DockArea
from enaml.workbench.ui.api import Workspace


class ExperimentWorkspace(Workspace):

    dock_area = Typed(DockArea)

    def start(self):
        self.content = Container()
        self.dock_area = DockArea(name='dock_area')
        self.dock_area.set_parent(self.content)
        deferred_call(self.plugins_started)

    def plugins_started(self):
        controller = self.workbench.get_plugin('psi.controller')
        controller.invoke_actions('plugins_started')
