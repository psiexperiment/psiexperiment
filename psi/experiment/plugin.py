from atom.api import Typed
from enaml.application import deferred_call
from enaml.layout.api import FloatItem, InsertItem
from enaml.layout.dock_layout import DockLayoutValidator
from enaml.workbench.plugin import Plugin
from enaml.widgets.api import Action, ToolBar
from enaml.widgets.toolkit_object import ToolkitObject

from .preferences import Preferences


TOOLBAR_POINT = 'psi.experiment.toolbar'
WORKSPACE_POINT = 'psi.experiment.workspace'
PREFERENCES_POINT = 'psi.experiment.preferences'


class MissingDockLayoutValidator(DockLayoutValidator):

    def result(self, node):
        return self._available - self._seen_items


class ExperimentPlugin(Plugin):

    _preferences = Typed(dict, {})

    def start(self):
        self._refresh_preferences()
        self._bind_observers()

    def setup_workspace(self, workspace):
        point = self.workbench.get_extension_point(WORKSPACE_POINT)
        for extension in point.extensions:
            extension.factory(self.workbench, workspace)

    def setup_toolbar(self, workspace):
        toolbars = []
        point = self.workbench.get_extension_point(TOOLBAR_POINT)
        for extension in point.extensions:
            children = extension.get_children(ToolkitObject)
            tb = ToolBar()
            tb.children.extend(children)
            toolbars.append(tb)
        workspace.toolbars = toolbars

    def get_layout(self):
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        return {'geometry': ui._window.geometry(),
                'dock_layout': ui.workspace.dock_area.save_layout()}

    def set_layout(self, layout):
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        ui._window.set_geometry(layout['geometry'])
        ui.workspace.dock_area.layout = layout['dock_layout']
        available = [i.name for i in ui.workspace.dock_area.dock_items()]
        missing = MissingDockLayoutValidator(available)(layout['dock_layout'])
        for item in missing:
            op = FloatItem(item=item)
            deferred_call(ui.workspace.dock_area.update_layout, op)

    def _refresh_preferences(self):
        preferences = {}
        point = self.workbench.get_extension_point(PREFERENCES_POINT)
        for extension in point.extensions:
            pref = extension.get_children(Preferences)[0]
            preferences[extension.plugin_id] = pref
        self._preferences = preferences

    def _bind_observers(self):
        self.workbench.get_extension_point(PREFERENCES_POINT) \
            .observe('extensions', self._refresh_preferences)

    def get_preferences(self):
        state = {}
        for plugin_id, preference in self._preferences.items():
            plugin = self.workbench.get_plugin(plugin_id)
            state[plugin_id] = preference.get_preferences(plugin)
        return state

    def set_preferences(self, state):
        for plugin_id, s in state.items():
            plugin = self.workbench.get_plugin(plugin_id)
            preference = self._preferences[plugin_id]
            preference.set_preferences(plugin, s)
