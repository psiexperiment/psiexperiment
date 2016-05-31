from atom.api import Unicode, Enum
from enaml.core.api import Declarative, d_


class Channel(Declarative):

    name = d_(Unicode())
    engine = d_(Unicode())
    channel = d_(Unicode())
    io_type = d_(Enum('hw_ao', 'hw_ai', 'hw_di', 'hw_do'))
    mode = d_(Enum('continuous', 'epoch', 'event'))
