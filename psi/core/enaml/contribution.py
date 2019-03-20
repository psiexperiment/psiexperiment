import logging
log = logging.getLogger(__name__)

from atom.api import Unicode
from enaml.core.api import Declarative, d_

from .util import load_manifest


class PSIContribution(Declarative):

    name = d_(Unicode())
    label = d_(Unicode())
    manifest = d_(Unicode())

    def _default_name(self):
        # Provide a default name if none is specified
        return self.parent.name + '.' + self.__class__.__name__

    @classmethod
    def find_manifest_class(cls):
        potential_locations = [
            f'{cls.__module__}.{cls.__name__}Manifest',
            f'{cls.__module__}_manifest.{cls.__name__}Manifest',
        ]
        for location in potential_locations:
            try:
                print(location)
                return load_manifest(location)
            except ImportError:
                pass

        m = f'Could not find manifest for {cls.__module__}.{cls.__name__}'
        raise ImportError(m)

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
        except ValueError:
            workbench.unregister(manifest.id)
            workbench.register(manifest)
