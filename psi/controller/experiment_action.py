from atom.api import Unicode
from enaml.core.api import Declarative, d_


class ExperimentAction(Declarative):

    # TODO: Create an ExperimentEvent object so that we can define the
    # available events that occur and can be subscribed to by actions.
    event = d_(Unicode())

    # Command
    command = d_(Unicode())
