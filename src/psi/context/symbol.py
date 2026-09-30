import importlib

from atom.api import Str, Callable
from enaml.core.api import Declarative, d_


class Symbol(Declarative):

    pass


class ImportedSymbol(Symbol):

    module = d_(Str())

    def get_object(self):
        return importlib.import_module(self.module)


class Function(Symbol):

    function = d_(Callable())

    def get_object(self):
        return self.function
