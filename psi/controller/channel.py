# TODO: Implement channel calibration. This is inherently tied to the engine
# though.

from atom.api import Unicode, Enum, Typed, Tuple
from enaml.core.api import Declarative, d_

from .engine import Engine

class Channel(Declarative):

    name = d_(Unicode())
    engine_name = d_(Unicode())
    io_type = d_(Enum('hw_ao', 'hw_ai', 'hw_di', 'sw_do'))
    channel = d_(Unicode())

    engine = Typed(Engine)
    fs = d_(Typed(object))
    start_trigger = d_(Unicode())
    expected_range = d_(Tuple)
