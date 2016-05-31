from atom.api import Typed, Unicode, Enum
from enaml.core.api import Declarative, d_


class Channel(Declarative):

    label = d_(Unicode())
    engine = d_(Unicode())
    channel = d_(Unicode())
    io_type = d_(Enum('hw_ao', 'hw_ai', 'hw_di', 'hw_do'))
    mode = d_(Enum('continuous', 'epoch', 'event'))

    _token_name = Unicode()
    _plugin_id = Unicode()
