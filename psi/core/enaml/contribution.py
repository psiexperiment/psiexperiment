import logging
log = logging.getLogger(__name__)

import importlib

from atom.api import Bool, Unicode

import enaml
from enaml.core.api import Declarative, d_


class PSIContribution(Declarative):

    name = d_(Unicode())
    label = d_(Unicode())
    manifest = d_(Unicode())

    def _default_name(self):
        # Provide a default name if none is specified
        return self.parent.name + '.' + self.__class__.__name__

    @classmethod
    def find_manifest_class(cls):
        with enaml.imports():
            for c in cls.mro():
                class_name = c.__name__ + 'Manifest'
                # First, check to see if the manifest is defined in the same
                # module as the contribution.
                try:
                    module = importlib.import_module(c.__module__)
                    return getattr(module, class_name)
                except AttributeError:
                    pass

                # Second, check to see if the manifest is defined in another,
                # appropriately-named, module.
                module_name = c.__module__ + '_manifest'
                try:
                    module = importlib.import_module(module_name)
                    manifest_class = getattr(module, class_name)
                    return manifest_class
                except ImportError as e:
                    pass
                except AttributeError:
                    pass

        raise ImportError

    def load_manifest(self, workbench):
        if not self.load_manifest:
            return
        try:
            manifest_class = self.find_manifest_class()
            manifest = manifest_class(contribution=self)
            workbench.register(manifest)
            m = 'Loaded manifest for contribution {} ({})'
            log.info(m.format(self.name, manifest_class.__name__))
        except ImportError:
            m = 'No manifest defind for contribution {}'
            log.warn(m.format(self.name))
