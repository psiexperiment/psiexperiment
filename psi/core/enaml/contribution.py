import importlib

from atom.api import Unicode
import enaml
from enaml.core.api import Declarative, d_


class PSIContribution(Declarative):
    
    name = d_(Unicode())
    label = d_(Unicode())
    manifest = d_(Unicode())

    def load_manifest(self):
        if self.manifest is not None:
            module_name, class_name = self.manifest.rsplit('.', 1)
            module = importlib.import_module(module_name)
            klass = getattr(module, class_name)            
            return klass(contribution=self)
