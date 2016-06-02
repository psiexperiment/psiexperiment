from atom.api import Unicode, Float, Bool
from enaml.core.api import Declarative, d_


class Engine(Declarative):

    name = d_(Unicode())
    ao_fs = d_(Float(100e3))
    ai_fs = d_(Float(25e3))
    master_clock = d_(Bool(False))

    def configure(self, configuration):
        raise NotImplementedError

    def register_ao_callback(self, callback):
        raise NotImplemneted

    def register_ai_callback(self, callback):
        raise NotImplemneted

    def register_et_callback(self, callback):
        raise NotImplemneted

    def start(self):
        raise NotImplementedError
