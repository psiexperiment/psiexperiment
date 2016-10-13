from atom.api import Unicode
from enaml.core.api import Declarative, d_


class CompatibleManifest(Declarative):

    id = d_(Unicode())


class RequiredManifest(Declarative):

    id = d_(Unicode())
