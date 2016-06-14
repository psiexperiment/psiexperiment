from functools import partial

from atom.api import Unicode, Float, Typed, Int, Property
from enaml.core.api import Declarative, d_


class Input(Declarative):

    channel = Property()
    engine = Property()

    def _get_channel(self):
        return self.parent

    def _get_engine(self):
        return self.parent.parent

    def configure(self, plugin):
        raise NotImplementedError


class ContinuousAnalogInput(Input):

    def configure(self, plugin):
        cb = partial(plugin.ai_callback, self.name)
        self.engine.register_ai_callback(cb, self.channel.name)


class AnalogThreshold(Input):

    threshold = d_(Float())
    debounce = d_(Int())

    def configure(self, plugin):
        cb = partial(plugin.et_callback, self.name)
        self.engine.register_ai_threshold_callback(cb, 
                                                   self.channel.name,
                                                   self.threshold,
                                                   self.debounce)
