import logging
log = logging.getLogger(__name__)

from atom.api import Typed
from enaml.application import deferred_call
from enaml.layout.api import FloatItem, InsertItem
from enaml.layout.dock_layout import DockLayoutValidator
from enaml.workbench.plugin import Plugin
from enaml.widgets.api import Action, ToolBar
from enaml.widgets.toolkit_object import ToolkitObject

from .preferences import Preferences
from .manifests import CompatibleManifest, RequiredManifest


TOOLBAR_POINT = 'psi.experiment.toolbar'
WORKSPACE_POINT = 'psi.experiment.workspace'
PREFERENCES_POINT = 'psi.experiment.preferences'
MANIFEST_POINT = 'psi.experiment.manifest'


class MissingDockLayoutValidator(DockLayoutValidator):

    def result(self, node):
        return self._available - self._seen_items


class ExperimentPlugin(Plugin):

    _preferences = Typed(dict, {})

    def start(self):
        self._refresh_preferences()
        self._bind_observers()

    def setup_workspace(self, workspace):
        log.debug('Setting up workspace')
        point = self.workbench.get_extension_point(WORKSPACE_POINT)
        for extension in point.extensions:
            extension.factory(self.workbench, workspace)
        log.debug('Done setting up workspace')

    def setup_toolbar(self, workspace):
        toolbars = []
        point = self.workbench.get_extension_point(TOOLBAR_POINT)
        for extension in point.extensions:
            children = extension.get_children(ToolkitObject)
            tb = ToolBar(name=extension.id)
            tb.children.extend(children)
            toolbars.append(tb)
        workspace.toolbars = toolbars

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
        ui = self.workbench.get_plugin('enaml.workbench.ui')
        ui._window.set_geometry(layout['geometry'])
        ui._window.set_geometry(layout['geometry'])
        self._set_toolbar_layout(ui.workspace.toolbars, layout['toolbars'])
        ui.workspace.dock_area.layout = layout['dock_layout']
        available = [i.name for i in ui.workspace.dock_area.dock_items()]
        missing = MissingDockLayoutValidator(available)(layout['dock_layout'])
        for item in missing:
            log.debug('{} missing from saved dock layout'.format(item))
            op = FloatItem(item=item)
            deferred_call(ui.workspace.dock_area.update_layout, op)

    def _refresh_preferences(self, event=None):
        preferences = {}
        point = self.workbench.get_extension_point(PREFERENCES_POINT)
        for extension in point.extensions:
            preference = extension.get_children(Preferences)[0]
            if preference.name in preferences:
                raise ValueError('Cannot reuse preference name')
            preferences[preference.name] = preference
        self._preferences = preferences

    def _bind_observers(self):
        self.workbench.get_extension_point(PREFERENCES_POINT) \
            .observe('extensions', self._refresh_preferences)

    def get_preferences(self):
        state = {}
        for name, preference in self._preferences.items():
            log.debug('Getting preferences for {}'.format(name))
            state[name] = preference.get_preferences(self.workbench)
        return state

    def set_preferences(self, state):
        for name, s in state.items():
            log.debug('Setting preferences for {}'.format(name))
            try:
                preference = self._preferences[name]
                preference.set_preferences(self.workbench, s)
            except KeyError as e:
                log.warn(e)
