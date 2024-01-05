from atom.api import Str, Value
from enaml.core.api import d_

from psi.core.enaml.api import PSIContribution


class StatusItem(PSIContribution):

    label = d_(Str())
    anchor = d_(Value())

    def _default_name(self):
        return self.valid_name(self.label)

    def _default_anchor(self):
        return self.children[0]
