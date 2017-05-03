from atom.api import List, Unicode, Value
from enaml.core.api import Declarative, d_, d_func


class Preferences(Declarative):

    name = d_(Unicode())

    @d_func
    def get_object(self, workbench):
        raise NotImplementedError

    @d_func
    def set_preferences(self, workbench, preferences):
        raise NotImplementedError

    @d_func
    def get_preferences(self, workbench):
        raise NotImplementedError


class _AutoPreferences(Preferences):

    auto_save = d_(List())

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


class ItemPreferences(_AutoPreferences):

    item = d_(Value())

    def get_object(self, workbench):
        return self.item


class PluginPreferences(_AutoPreferences):

    plugin_id = d_(Unicode())

    def get_object(self, workbench):
        return workbench.get_plugin(self.plugin_id)
