import logging
log = logging.getLogger(__name__)

import re

from atom.api import Str
from enaml.core.api import Declarative, d_

from .util import load_manifest


class PSIContribution(Declarative):

    name = d_(Str())
    label = d_(Str())
    manifest = d_(Str())

    def _default_name(self):
        # Provide a default name if none is specified
        return self.parent.name + '.' + self.__class__.__name__

    @classmethod
    def valid_name(self, label):
        return re.sub('\W|^(?=\d)', '_', label)

    @classmethod
    def find_manifest_class(cls):
        search = []
        for c in cls.mro():
            search.append(f'{c.__module__}.{c.__name__}Manifest')
            search.append(f'{c.__module__}_manifest.{c.__name__}Manifest')
        search.append('psi.core.enaml.manifest.PSIManifest')
        for location in search:
            try:
                return load_manifest(location)
            except ImportError:
                pass

        # I'm not sure this can actually happen anymore since it should return
        # the base `PSIManifest` class at a minimum.
        m = f'Could not find manifest for {cls.__module__}.{cls.__name__}'
        raise ImportError(m)

    def load_manifest(self, workbench):
        #if not self.load_manifest:
        #    return
        try:
            manifest_class = self.find_manifest_class()
            manifest = manifest_class(contribution=self)
            workbench.register(manifest)
            m = 'Loaded manifest for contribution %s (%s) with ID %r'
            log.debug(m, self.name, manifest_class.__name__, manifest.id)
        except ImportError:
            m = 'No manifest defind for contribution %s'
            log.warn(m, self.name)
        except ValueError as e:
            m = f'Manifest "{manifest.id}" for plugin "{self.name}" already registered.'
            raise ImportError(m) from e

