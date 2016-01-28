from atom.api import Atom, Unicode, Typed, Value, List, Property, Bool
from expression import Expr


class Parameter(Atom):

    # These values could theoretically change, but I prefer that they remain
    # static throughout the lifetime of the application.
    name = Unicode()
    label = Unicode()
    dtype = Typed(type)
    default_value = Value()

    # These values can be changed in the GUI
    expression = Unicode()
    rove = Bool(False)

    def __init__(self, name, dtype, **kwargs):
        kwargs.setdefault('default_value', dtype())
        super(Parameter, self).__init__(name=name, dtype=dtype, **kwargs)
