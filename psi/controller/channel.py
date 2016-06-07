# TODO: Implement channel calibration. This is inherently tied to the engine
# though.

from atom.api import Unicode, Enum, Typed
from enaml.core.api import Declarative, d_

from .engine import Engine

class Channel(Declarative):

    name = d_(Unicode())
    engine_name = d_(Unicode())
    io_type = d_(Enum('hw_ao', 'hw_ai', 'hw_di', 'hw_do', 'et'))
    mode = d_(Enum('continuous', 'epoch', 'event'))
    channel = d_(Unicode())

    engine = Typed(Engine)
