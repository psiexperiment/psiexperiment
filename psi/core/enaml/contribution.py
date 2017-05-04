import logging
log = logging.getLogger(__name__)

import importlib

from atom.api import Unicode

import enaml
from enaml.core.api import Declarative, d_


class PSIContribution(Declarative):
    
    name = d_(Unicode())
    label = d_(Unicode())
    manifest = d_(Unicode())

    def load_manifest(self, workbench):
        if self.manifest and self.name:
            with enaml.imports():
                module_name, class_name = self.manifest.rsplit('.', 1)
                module = importlib.import_module(module_name)
                manifest_class = getattr(module, class_name)            
                manifest = manifest_class(contribution=self)
            workbench.register(manifest)
        else:
            m = 'No manifest defind for contribution {}'
            log.info(m.format(self.name))
