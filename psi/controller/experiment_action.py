import logging
log = logging.getLogger(__name__)

from atom.api import Unicode, Int, Dict, Bool, Typed
from enaml.core.api import Declarative, d_
from enaml.qt.QtCore import QRunnable


class ExperimentState(Declarative):
    '''
    Allows for indication of a state (e.g., `experiment_active`, `iti_active`).
    Automatically contributes the start/end events associataed with the state
    (e.g., `experiment_start`, `experiment_end`).
    '''
    name = d_(Unicode())
    active = Bool(False)
    events = ['prepare', 'start', 'end']

    def _generate_events(self):
        events = []
        for name in self.events:
            event_name = '{}_{}'.format(self.name, name)
            event = ExperimentEvent(name=event_name, associated_state=self)
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
            elif self.name.endswith('end'):
                self.associated_state.active = False
        self.active = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False


class ExperimentAction(Declarative):

    # Name of event that triggers command
    event = d_(Unicode())

    # Command to invoke
    command = d_(Unicode())

    # Arguments to pass to command by keywod
    kwargs = d_(Dict())

    # Should the action be invoked in its own thread?
    concurrent = d_(Bool(False))

    # Defines order of invocation. Less than 100 invokes before default. Higher
    # than 100 invokes after default. Note that if concurrent is True, then
    # order of execution is not guaranteed.
    weight = d_(Int(50))

    def match(self, context):
        return eval(self.event, context)


class QExperimentActionTask(QRunnable):

    def __init__(self, method):
        super(QExperimentActionTask, self).__init__()
        self.method = method

    def run(self):
        log.debug('Running action in remote thread')
        self.method()
