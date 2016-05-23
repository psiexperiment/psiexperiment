from atom.api import Typed, Unicode
from enaml.core.api import Declarative, d_

from psi.token import TokenManifest


class Output(Declarative):

    label = d_(Unicode())
    engine = d_(Unicode())
    channel = d_(Unicode())
    scope = d_(Unicode())
    _plugin_id = Unicode()


class Continuous(Output):
    pass


class Epoch(Output):

    reference = d_(Unicode())
