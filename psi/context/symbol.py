import importlib

from atom.api import Unicode, Callable
from enaml.core.api import Declarative, d_


class Symbol(Declarative):

    pass


class ImportedSymbol(Symbol):

    module = d_(Unicode())

    def get_object(self):
        return importlib.import_module(self.module)


class Function(Symbol):

    function = d_(Callable())

    def get_object(self):
        return self.function
