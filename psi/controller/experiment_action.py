import logging
log = logging.getLogger(__name__)

from atom.api import Unicode, Int, Dict, Bool, Typed, Callable
from enaml.core.api import Declarative, d_

from psi.util import get_dependencies


class ExperimentState(Declarative):
    '''
    Allows for indication of a state (e.g., `experiment_active`, `iti_active`).
    Automatically contributes the start/end events associataed with the state
    (e.g., `experiment_start`, `experiment_end`).
    '''
    name = d_(Unicode())
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
    associated_state = Typed(ExperimentState)


class ExperimentAction(Declarative):

    # Name of event that triggers command
    event = d_(Unicode())

    match = Callable()

    _code = Typed(object)
    _dependencies = Typed(object)
    _key = Typed(object)

    # Command to invoke
    command = d_(Unicode())

    # Arguments to pass to command by keywod
    kwargs = d_(Dict())

    # Defines order of invocation. Less than 100 invokes before default. Higher
    # than 100 invokes after default. Note that if concurrent is True, then
    # order of execution is not guaranteed.
    weight = d_(Int(50))

    def _observe_event(self, event):
        self._code = compile(self.event, 'dynamic', 'eval')
        self._dependencies = get_dependencies(self.event)
        if len(self._dependencies) == 1:
            self.match = self._match_simple
            self._key = self._dependencies[0]
        else:
            self.match = self._match_eval

    def _match_eval(self, context):
        return eval(self._code, context)

    def _match_simple(self, context):
        return context[self._key]
