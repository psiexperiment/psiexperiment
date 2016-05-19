from enaml.core.declarative import Declarative, d_
from atom.api import Unicode, Typed


class Result(Declarative):
    name = d_(Unicode())
    label = d_(Unicode())
    dtype = d_(Typed(type))
    group = d_(Unicode())
    compact_label = d_(Unicode())
