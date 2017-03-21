from atom.api import Typed

import enaml
from enaml.application import deferred_call
from enaml.layout.api import InsertItem
from enaml.widgets.api import DockItem, ToolBar
from enaml.workbench.ui.api import Workspace

with enaml.imports():
    from .view import ExperimentView


class ExperimentWorkspace(Workspace):

    toolbars = Typed(list, {})

    @property
    def dock_area(self):
        return self.content.find('dock_area')

    def start(self):
        self.content = ExperimentView()
        plugin = self.workbench.get_plugin('psi.experiment')
        core = self.workbench.get_plugin('enaml.workbench.core')
