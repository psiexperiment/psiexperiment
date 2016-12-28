from atom.api import Unicode, Int, Dict
from enaml.core.api import Declarative, d_
import code


class ExperimentState(Declarative):
    '''
    Allows for indication of a state (e.g., `experiment_active`, `iti_active`).
    Automatically contributes the start/end events associataed with the state
    (e.g., `experiment_start`, `experiment_stop`).
    '''
    name = d_(Unicode())
    active = Bool(False)
    events = ['prepare', 'start', 'end']

    def _generate_events(self):
        events = []
        for name in self.events:
            name = '{}_{}'.format(self.name, e)
            event = ExperimentEvent(name=name, associated_state=self)
            events.append(event)
        return events


class ExperimentEvent(Declarative):

    name = d_(Unicode())
    active = Bool(False)
    associated_state = Typed(ExperimentState)

    def __enter__(self):
        if self.associated_state is not None:
            if self.name.endswith('start'):
                self.associated_state.active = True
            elif self.name.endswith('end')
                self.associated_state.active = False
        self.active = True

    def __exit__(self):
        self.active = False


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

    def match(self, context):
        return eval(self.event, globals=None, locals=context)
