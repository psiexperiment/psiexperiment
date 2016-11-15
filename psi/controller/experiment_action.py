from atom.api import Unicode, Int, Dict
from enaml.core.api import Declarative, d_


class ExperimentEvent(Declarative):
    # TODO: Create an ExperimentEvent object so that we can define the
    # available events that occur and can be subscribed to by actions. This
    # will minimize typos.
    name = d_(Unicode())
    label = d_(Unicode())


class ExperimentAction(Declarative):

    # Name of event that triggers command
    event = d_(Unicode())

    # Command to invoke
    command = d_(Unicode())

    # Arguments to pass to command
    kwargs = d_(Dict())

    # Defines order of invocation. Less than 100 invokes before default. Higher
    # than 100 invokes after default.
    weight = d_(Int(100))
