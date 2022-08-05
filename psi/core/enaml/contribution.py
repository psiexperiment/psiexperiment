import logging
log = logging.getLogger(__name__)

import importlib.util
import re

from atom.api import Bool, Str
import enaml
from enaml.core.api import Declarative, d_


MANIFEST_CACHE = {}
SEARCH_CACHE = {}


class ManifestNotFoundError(ImportError):
    pass


def load_manifest(manifest_path):
    module_name, manifest_name = manifest_path.rsplit('.', 1)
    with enaml.imports():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name != module_name:
                raise
            raise ManifestNotFoundError() from e
    try:
        return getattr(module, manifest_name)
    except AttributeError as e:
        raise ManifestNotFoundError() from e


def find_manifest_class(obj):
    cls = obj.__class__
    if cls in MANIFEST_CACHE:
        return MANIFEST_CACHE[cls]

    search = []
    for c in cls.mro():
        base_module = c.__module__.split('.', 1)[0]
        if base_module in ('atom', 'enaml', 'builtins'):
            continue
        search.append(f'{c.__module__}.{c.__name__}Manifest')
        search.append(f'{c.__module__}_manifest.{c.__name__}Manifest')
    search.append('psi.core.enaml.manifest.PSIManifest')
    log.debug(f'Attempting to locate manifest for %s from candidates %s',
                cls.__name__, '\n ... '.join([''] + search))
    for location in search:
        if location in SEARCH_CACHE:
            if SEARCH_CACHE[location] is not None:
                MANIFEST_CACHE[cls] = manifest = SEARCH_CACHE[location]
                return manifest
            else:
                continue
        try:
            MANIFEST_CACHE[cls] = manifest = load_manifest(location)
            SEARCH_CACHE[location] = manifest
            log.debug('... Found manifest at %s', location)
            return manifest
        except ManifestNotFoundError as e:
            SEARCH_CACHE[location] = None

    # I'm not sure this can actually happen anymore since it should return
    # the base `PSIManifest` class at a minimum.
    m = f'Could not find manifest for {cls.__module__}.{cls.__name__}'
    raise ManifestNotFoundError(m)


class PSIContribution(Declarative):

    #: Name of contribution. Should be unique among all contributions for that
    #: subclass as it is typically used by the plugins to find a particular
    #: contribution.
    name = d_(Str()).tag(metadata=True)

    #: Label of contribution. Not required, but a good idea as it affects how
    #: the contribution may be represented in the user interface.
    label = d_(Str()).tag(metadata=True)

    #: Flag indicating whether item has already been registered.
    registered = Bool(False)

    def _default_name(self):
        # Provide a default name if none is specified TODO: make this mandatory
        # (i.e., no default?)
        return self.parent.name + '.' + self.__class__.__name__

    def _default_label(self):
        return self.name.replace('_', ' ')

    @classmethod
    def valid_name(self, label):
        '''
        Can be used to convert a label provided to a valid name
        '''
        # TODO: Get rid of this?
        return re.sub('\W|^(?=\d)', '_', label)

    def find_manifest_class(self):
        return find_manifest_class(self)

    def load_manifest(self, workbench):
        if self.registered:
            return
        try:
            manifest_class = self.find_manifest_class()
            manifest = manifest_class(contribution=self)
            workbench.register(manifest)
            self.registered = True
            m = 'Loaded manifest for contribution %s (%s) with ID %r'
            log.debug(m, self.name, manifest_class.__name__, manifest.id)
        except ManifestNotFoundError:
            m = 'No manifest defined for contribution %s'
            log.warn(m, self.name)
        except ValueError as e:
            m = f'Manifest "{manifest.id}" for plugin "{self.name}" already registered.'
            raise ImportError(m) from e
