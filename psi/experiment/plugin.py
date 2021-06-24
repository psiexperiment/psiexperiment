import logging
log = logging.getLogger(__name__)

from atom.api import Typed
from enaml.application import deferred_call
from enaml.layout.api import InsertItem
from enaml.layout.dock_layout import DockLayoutValidator
from enaml.workbench.plugin import Plugin
from enaml.widgets.api import Action, DockItem, ToolBar
from enaml.widgets.toolkit_object import ToolkitObject

from .preferences import Preferences
from .status_item import StatusItem


TOOLBAR_POINT = 'psi.experiment.toolbar'
WORKSPACE_POINT = 'psi.experiment.workspace'
STATUS_POINT = 'psi.experiment.status'
PREFERENCES_POINT = 'psi.experiment.preferences'


# TODO: There's a bizzare bug in the warnings system that causes the global
# namespace for the module invoking the validator to be cleared.
DockLayoutValidator.warn = lambda *args, **kw: None


class MissingDockLayoutValidator(DockLayoutValidator):

    def result(self, node):
        return self._available - self._seen_items

    def warn(self, message):
        return


class ExperimentPlugin(Plugin):

    _preferences = Typed(list)
    _workspace_contributions = Typed(list)
    _toolbars = Typed(list)
    _status_items = Typed(list)

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
        log.debug('Refreshing workspace')
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        point = self.workbench.get_extension_point(WORKSPACE_POINT)
        items_added = []
        for extension in point.extensions:
            if extension.factory is not None:
                items = extension.factory(self, ui.workbench, ui.workspace)
            else:
                items = []
            items.extend(extension.get_children(DockItem))
            for item in items:
                item.set_parent(ui.workspace.dock_area)
                op = InsertItem(item=item.name)
                ui.workspace.dock_area.update_layout(op)
                items_added.append(f'{item.name} ({item.title}) from {extension.id}')
        log.debug('Added dock area items: %s', ', '.join(items_added))

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
        self._toolbars = toolbars

    def _refresh_preferences(self, event=None):
        log.debug('Refreshing preferences')
        preferences = []
        names = []
        point = self.workbench.get_extension_point(PREFERENCES_POINT)
        preferences_added = []
        for extension in point.extensions:
            for preference in extension.get_children(Preferences):
                if preference.name in names:
                    raise ValueError('Cannot reuse preference name')
                preferences.append(preference)
                preferences_added.append(preference.name)
        log.debug('Registered preferences: %s', ', '.join(preferences_added))
        self._preferences = preferences

    def _refresh_status(self, event=None):
        log.debug('Refreshing status')
        point = self.workbench.get_extension_point(STATUS_POINT)
        status_items_added = []
        for extension in point.extensions:
            for item in extension.get_children(StatusItem):
                item.load_manifest(self.workbench)
                status_items_added.append(item)
        self._status_items = status_items_added

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
                #toolbar.proxy.widget.setFloating(floating)
                #toolbar.proxy.widget.move(x, y)

    def get_layout(self):
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        return {'geometry': ui._window.geometry(),
                'toolbars': self._get_toolbar_layout(ui.workspace.toolbars),
                'dock_layout': ui.workspace.dock_area.save_layout()}

    def set_layout(self, layout):
        log.debug('Setting layout')
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        try:
            ui._window.set_geometry(layout['geometry'])
        except Exception as e:
            log.exception(e)

        try:
            self._set_toolbar_layout(ui.workspace.toolbars, layout['toolbars'])
        except Exception as e:
            log.exception(e)

        available = [i.name for i in ui.workspace.dock_area.dock_items()]
        missing = MissingDockLayoutValidator(available)(layout['dock_layout'])
        for item in missing:
            log.debug('{} missing from saved dock layout'.format(item))
            op = InsertItem(item=item)
            deferred_call(ui.workspace.dock_area.update_layout, op)
        ui.workspace.dock_area.layout = layout['dock_layout']

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
