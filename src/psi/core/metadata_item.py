from atom.api import Str
from enaml.core.api import d_
from psi.core.enaml.api import PSIContribution


class MetadataItem(PSIContribution):

    value = d_(Str())
