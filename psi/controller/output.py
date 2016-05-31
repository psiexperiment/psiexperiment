from atom.api import Unicode, Enum, Typed
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Plugin


class Output(Declarative):
    
    label = d_(Unicode())
    name = d_(Unicode())
    channel = d_(Unicode())
    mode = d_(Enum('continuous', 'epoch'))

    _token_name = Unicode()
    _plugin_id = Unicode()
    _plugin = Typed(Plugin)

    def get_waveform(self, offset, samples):
        return self._plugin.get_waveform(offset, samples)
