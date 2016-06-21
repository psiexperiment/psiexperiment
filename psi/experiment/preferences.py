from atom.api import List, Unicode
from enaml.core.api import Declarative, d_, d_func


class Preferences(Declarative):

    auto_save = d_(List())
    name = d_(Unicode())

    @d_func
    def get_object(self, workbench):
        raise NotImplementedError

    @d_func
    def set_preferences(self, workbench, preferences):
        obj = self.get_object(workbench)
        for m, v in preferences.items():
            setattr(obj, m, v)

    @d_func
    def get_preferences(self, workbench):
        obj = self.get_object(workbench)
        return {m: getattr(obj, m) for m in self.auto_save}


class PluginPreferences(Preferences):

    plugin_id = d_(Unicode())

    def get_object(self, workbench):
        return workbench.get_plugin(self.plugin_id)

class UIPreferences(Preferences):

    item_name = d_(Unicode())

    @d_func
    def get_object(self, workbench):
        ui = workbench.get_plugin('enaml.workbench.ui')
        return ui.workspace.content.find(self.item_name)
