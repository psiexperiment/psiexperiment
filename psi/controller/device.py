from atom.api import Unicode
from enaml.core.api import Declarative, d_

class Device(Declarative):

    name = d_(Unicode())
    label = d_(Unicode())

    def load_manifest(self):
        return None
