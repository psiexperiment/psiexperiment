import enaml
from enaml.application import deferred_call
from enaml.layout.api import InsertItem
from enaml.widgets.api import DockItem
from enaml.workbench.ui.api import Workspace

with enaml.imports():
    from .view import ExperimentView


class ExperimentWorkspace(Workspace):

    @property
    def dock_area(self):
        return self.content.find('dock_area')

    @property
    def toolbar(self):
        return self.content.find('tool_bar')

    def start(self):
        self.content = ExperimentView(workspace=self)
        plugin = self.workbench.get_plugin('psi.experiment')
        plugin.setup_toolbar(self)
        plugin.setup_workspace(self)
        core = self.workbench.get_plugin('enaml.workbench.core')
        deferred_call(core.invoke_command, 'psi.get_default_layout')
        deferred_call(core.invoke_command, 'psi.get_default_preferences')
        deferred_call(core.invoke_command, 'psi.get_default_context')
