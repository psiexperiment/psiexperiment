import logging
log = logging.getLogger(__name__)

from atom.api import Typed
from enaml.application import deferred_call
from enaml.layout.api import FloatItem
from enaml.layout.dock_layout import DockLayoutValidator
from enaml.workbench.plugin import Plugin
from enaml.widgets.api import Action, ToolBar, DockItem
from enaml.widgets.toolkit_object import ToolkitObject

from .preferences import Preferences


TOOLBAR_POINT = 'psi.experiment.toolbar'
WORKSPACE_POINT = 'psi.experiment.workspace'
PREFERENCES_POINT = 'psi.experiment.preferences'


class MissingDockLayoutValidator(DockLayoutValidator):

    def result(self, node):
        return self._available - self._seen_items


class ExperimentPlugin(Plugin):

    _preferences = Typed(list)
    _workspace_contributions = Typed(list)
    _toolbar_contributions = Typed(list)

    def start(self):
        log.debug('Starting experiment plugin')
        self._refresh_workspace()
        self._refresh_toolbars()
        self._refresh_preferences()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _refresh_workspace(self, event=None):
        log.debug('Refreshing workspace')
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        point = self.workbench.get_extension_point(WORKSPACE_POINT)
        for extension in point.extensions:
            if extension.factory is not None:
                extension.factory(ui.workbench, ui.workspace)
            for item in extension.get_children(DockItem):
                if hasattr(item, 'plugin'):
                    plugin = self.workbench.get_plugin(extension.parent.id)
                    item.plugin = plugin
                item.set_parent(ui.workspace.dock_area)
                op = FloatItem(item=item.name)
                deferred_call(ui.workspace.dock_area.update_layout, op)

    def _refresh_toolbars(self, event=None):
        log.debug('Refreshing toolbars')
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        toolbars = []
        point = self.workbench.get_extension_point(TOOLBAR_POINT)
        for extension in point.extensions:
            children = extension.get_children(ToolkitObject)
            tb = ToolBar(name=extension.id)
            tb.children.extend(children)
            toolbars.append(tb)
        ui.workspace.toolbars = toolbars

    def _refresh_preferences(self, event=None):
        log.debug('Refreshing preferences')
        preferences = []
        names = []
        point = self.workbench.get_extension_point(PREFERENCES_POINT)
        for extension in point.extensions:
            for preference in extension.get_children(Preferences):
                if preference.name in names:
                    raise ValueError('Cannot reuse preference name')
                preferences.append(preference)
                log.debug('Registering preference {}'.format(preference.name))
        self._preferences = preferences

    def _bind_observers(self):
        self.workbench.get_extension_point(PREFERENCES_POINT) \
            .observe('extensions', self._refresh_preferences)
        self.workbench.get_extension_point(TOOLBAR_POINT) \
            .observe('extensions', self._refresh_toolbars)
        self.workbench.get_extension_point(WORKSPACE_POINT) \
            .observe('extensions', self._refresh_workspace)

    def _unbind_observers(self):
        self.workbench.get_extension_point(PREFERENCES_POINT) \
            .unobserve('extensions', self._refresh_preferences)
        self.workbench.get_extension_point(TOOLBAR_POINT) \
            .unobserve('extensions', self._refresh_toolbars)
        self.workbench.get_extension_point(WORKSPACE_POINT) \
            .unobserve('extensions', self._refresh_workspace)

    def _get_toolbar_layout(self, toolbars):
        # TODO: This needs some work. It's not *quite* working 100%, especially
        # when docked.
        layout = {}
        for toolbar in toolbars:
            x = toolbar.proxy.widget.x()
            y = toolbar.proxy.widget.y()
            floating = toolbar.proxy.widget.isFloating()
            layout[toolbar.name] = x, y, floating
        return layout

    def _set_toolbar_layout(self, toolbars, layout):
        for toolbar in toolbars:
            if toolbar.name in layout:
                x, y, floating = layout[toolbar.name]
                toolbar.proxy.widget.setFloating(floating)
                toolbar.proxy.widget.move(x, y)

    def get_layout(self):
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        return {'geometry': ui._window.geometry(),
                'toolbars': self._get_toolbar_layout(ui.workspace.toolbars),
                'dock_layout': ui.workspace.dock_area.save_layout()}

    def set_layout(self, layout):
        log.debug('Setting layout')
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        ui._window.set_geometry(layout['geometry'])
        self._set_toolbar_layout(ui.workspace.toolbars, layout['toolbars'])
        ui.workspace.dock_area.layout = layout['dock_layout']
        available = [i.name for i in ui.workspace.dock_area.dock_items()]
        missing = MissingDockLayoutValidator(available)(layout['dock_layout'])
        for item in missing:
            log.debug('{} missing from saved dock layout'.format(item))
            op = FloatItem(item=item)
            deferred_call(ui.workspace.dock_area.update_layout, op)

    def _bind_observers(self):
        self.workbench.get_extension_point(PREFERENCES_POINT) \
            .observe('extensions', self._refresh_preferences)

    def get_preferences(self):
        state = {}
        for preference in self._preferences:
            log.debug('Getting preferences for {}'.format(preference.name))
            state[preference.name] = preference.get_preferences(self.workbench)
        return state

    def set_preferences(self, state):
        for preference in self._preferences:
            log.debug('Setting preferences for {}'.format(preference.name))
            if preference.name not in state:
                log.warn('Preference {} missing'.format(preference.name))
            else:
                s = state[preference.name]
                preference.set_preferences(self.workbench, s)
