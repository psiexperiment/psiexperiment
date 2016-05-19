from atom.api import List
from enaml.core.api import Declarative, d_, d_func


class Preferences(Declarative):

    auto_save = d_(List())

    @d_func
    def get_object(self, plugin):
        return plugin

    @d_func
    def get_preferences(self, plugin):
        obj = self.get_object(plugin)
        return dict((m, getattr(obj, m)) for m in self.auto_save)

    @d_func
    def set_preferences(self, plugin, preferences):
        obj = self.get_object(plugin)
        for m, v in preferences.items():
            setattr(obj, m, v)
