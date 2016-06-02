from atom.api import Unicode, Enum, Typed
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Plugin

from .engine import Engine
from .channel import Channel


class Output(Declarative):

    label = d_(Unicode())
    name = d_(Unicode())
    channel_name = d_(Unicode())
    mode = d_(Enum('continuous', 'epoch'))

    _token_name = Unicode()
    _plugin_id = Unicode()
    _plugin = Typed(Plugin)

    channel = Typed(Channel)

    def get_waveform(self, offset=0, samples=None):
        if self.mode == 'continuous' and samples is None:
            raise ValueError('Must specify the number of samples')
        return self._plugin.get_waveform(offset, samples)
