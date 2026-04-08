from atom.api import Bool, Str, Value
from enaml.core.api import d_

from psi.core.enaml.api import PSIContribution


class StatusItem(PSIContribution):

    label = d_(Str())
    anchor = d_(Value())

    #: Attach trailing spacer to ensure that items aren't forced to fill full
    #: width. Can set to False for progressbars. 
    trailing_spacer = d_(Bool(True))

    def _default_name(self):
        return self.valid_name(self.label)

    def _default_anchor(self):
        return self.children[0]
