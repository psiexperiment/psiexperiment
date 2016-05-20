from atom.api import Unicode, Bool
from enaml.core.api import Declarative, d_


class ContextGroup(Declarative):
    '''
    Used to group together context items for management.
    '''
    # Group name
    name = d_(Unicode())

    # Label to use in the GUI
    label = d_(Unicode())

    # Are the parameters in this group visible?
    visible = d_(Bool(True))
