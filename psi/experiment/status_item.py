from atom.api import Str
from enaml.core.api import d_

from psi.core.enaml.api import PSIContribution


class StatusItem(PSIContribution):

    label = d_(Str())

    def _default_name(self):
        return self.valid_name(self.label)
