from functools import partial

from atom.api import Unicode, Enum, Typed, Property
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Plugin

class Output(Declarative):

    label = d_(Unicode())
    name = d_(Unicode())

    channel = Property()
    engine = Property()

    # TODO: clean this up. it's sort of hackish.
    _token_name = Unicode()
    _plugin_id = Unicode()
    _plugin = Typed(Plugin)

    def _get_channel(self):
        return self.parent

    def _get_engine(self):
        return self.parent.parent


class EpochOutput(Output):

    def get_waveform(self, offset=0, samples=None):
        return self._plugin.get_waveform(offset, samples)

    def configure(self, plugin):
        self._plugin.initialize(self.channel.fs)


class ContinuousOutput(Output):

    def get_waveform(self, offset, samples):
        return self._plugin.get_waveform(offset, samples)

    def configure(self, plugin):
        cb = partial(plugin.ao_callback, self.name)
        self.engine.register_ao_callback(cb, self.channel.name)
        self._plugin.initialize(self.channel.fs)
