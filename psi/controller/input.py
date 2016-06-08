from atom.api import Unicode, Enum, Typed
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Plugin

from .engine import Engine
from .channel import Channel


class Input(Declarative):

    label = d_(Unicode())
    name = d_(Unicode())
    channel_name = d_(Unicode())
    mode = d_(Enum('continuous', 'epoch'))

    channel = Typed(Channel)
