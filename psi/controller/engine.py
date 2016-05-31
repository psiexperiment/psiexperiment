from enaml.core.api import Declarative, d_
from atom.api import Unicode


class Engine(Declarative):

    name = d_(Unicode())

    def configure(self, configuration):
        raise NotImplementedError
