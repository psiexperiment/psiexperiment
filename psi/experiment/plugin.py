import logging
log = logging.getLogger(__name__)

import textwrap

from atom.api import Typed
from enaml.application import deferred_call
from enaml.layout.api import InsertItem
from enaml.layout.dock_layout import DockLayoutValidator
from enaml.workbench.plugin import Plugin
from enaml.widgets.api import Action, DockItem, ToolBar
from enaml.widgets.toolkit_object import ToolkitObject

from psi.core.enaml.api import PSIPlugin
from .preferences import Preferences
from .status_item import StatusItem


TOOLBAR_POINT = 'psi.experiment.toolbar'
WORKSPACE_POINT = 'psi.experiment.workspace'
STATUS_POINT = 'psi.experiment.status'
PREFERENCES_POINT = 'psi.experiment.preferences'


class MissingDockLayoutValidator(DockLayoutValidator):

    def result(self, node):
        return self._available - self._seen_items


class ExperimentPlugin(PSIPlugin):

    _preferences = Typed(dict)
    _workspace_contributions = Typed(dict)
    _toolbars = Typed(dict, {})
    _status_items = Typed(dict)

    def start(self):
        log.debug('Starting experiment plugin')
        self._refresh_status()
        self._refresh_workspace()
        self._refresh_toolbars()
        self._refresh_preferences()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _refresh_workspace(self, event=None):
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        items = self.load_plugins(WORKSPACE_POINT, DockItem, 'name',
                                  plugin=self)

        ops = []
        for item in items.values():
            item.set_parent(ui.workspace.dock_area)
            ops.append(InsertItem(item=item.name))

        deferred_call(ui.workspace.dock_area.update_layout, ops)
        self._workspace_contributions = items

    def _refresh_toolbars(self, event=None):
        log.debug('Refreshing toolbars')
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        toolbars = {}
        point = self.workbench.get_extension_point(TOOLBAR_POINT)
        for extension in point.extensions:
            children = extension.get_children(ToolkitObject)
            toolbars[extension.id] = tb = ToolBar(name=extension.id)
            tb.children.extend(children)
        self._toolbars = toolbars

    def _refresh_preferences(self, event=None):
        preferences = self.load_plugins(PREFERENCES_POINT, Preferences, 'name')
        self._preferences = preferences

    def _refresh_status(self, event=None):
        status_items = self.load_plugins(STATUS_POINT, StatusItem, 'label')
        self._status_items = status_items

    def _bind_observers(self):
        self.workbench.get_extension_point(PREFERENCES_POINT) \
            .observe('extensions', self._refresh_preferences)
        self.workbench.get_extension_point(TOOLBAR_POINT) \
            .observe('extensions', self._refresh_toolbars)
        self.workbench.get_extension_point(WORKSPACE_POINT) \
            .observe('extensions', self._refresh_workspace)
        self.workbench.get_extension_point(STATUS_POINT) \
            .observe('extensions', self._refresh_status)

    def _unbind_observers(self):
        self.workbench.get_extension_point(PREFERENCES_POINT) \
            .unobserve('extensions', self._refresh_preferences)
        self.workbench.get_extension_point(TOOLBAR_POINT) \
            .unobserve('extensions', self._refresh_toolbars)
        self.workbench.get_extension_point(WORKSPACE_POINT) \
            .unobserve('extensions', self._refresh_workspace)
        self.workbench.get_extension_point(STATUS_POINT) \
            .unobserve('extensions', self._refresh_status)

    def _get_toolbar_layout(self):
        # TODO: This needs some work. It's not *quite* working 100%, especially
        # when docked.
        layout = {}
        for name, toolbar in self._toolbars.items():
            layout[name] = {
                'floating': toolbar.floating,
                'orientation': toolbar.orientation,
                'dock_area': toolbar.dock_area,
                'x': toolbar.proxy.widget.x(),
                'y': toolbar.proxy.widget.y(),
            }
        return layout

    def _set_toolbar_layout(self, layout):
        for name, toolbar in self._toolbars.items():
            if name in layout:
                toolbar.floating = layout[name].get('floating', False)
                toolbar.orientation = layout[name].get('orientation', 'horizontal')
                toolbar.dock_area = layout[name].get('dock_area', 'top')
                toolbar.proxy.widget.move(layout[name].get('x', 0),
                                          layout[name].get('y', 0))

    def get_layout(self):
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        return {'geometry': ui._window.geometry(),
                'toolbars': self._get_toolbar_layout(),
                'dock_layout': ui.workspace.dock_area.save_layout()}

    def set_layout(self, layout):
        log.warning('Setting layout')
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        try:
            ui._window.set_geometry(layout['geometry'])
        except Exception as e:
            log.exception(e)

        try:
            self._set_toolbar_layout(layout['toolbars'])
        except Exception as e:
            log.exception(e)

        available = [i.name for i in ui.workspace.dock_area.dock_items()]
        missing = MissingDockLayoutValidator(available)(layout['dock_layout'])
        ops = []
        for item in missing:
            log.debug('{} missing from saved dock layout'.format(item))
            ops.append(InsertItem(item=item))
        ui.workspace.dock_area.layout = layout['dock_layout']
        deferred_call(ui.workspace.dock_area.update_layout, ops)

    def _bind_observers(self):
        self.workbench.get_extension_point(PREFERENCES_POINT) \
            .observe('extensions', self._refresh_preferences)

    def get_preferences(self):
        state = {}
        for name, preference in self._preferences.items():
            log.debug('Getting preferences for %s', name)
            state[name] = preference.get_preferences(self.workbench)
        return state

    def set_preferences(self, state):
        for name, preference in self._preferences.items():
            log.debug('Setting preferences for %s', name)
            if name not in state:
                log.warn('Preference %s missing', name)
            else:
                preference.set_preferences(self.workbench, state[name])
