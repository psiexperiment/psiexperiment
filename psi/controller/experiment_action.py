import logging
log = logging.getLogger(__name__)

from functools import partial

from atom.api import Unicode, Int, Dict, Bool, Typed, Callable, List
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


missing_event_mesg = '''
Missing event "{key}".

Perhaps an input, output or device is missing from the IO configuration?
'''


def simple_match(key, context):
    try:
        return context[key]
    except Exception as e:
        new_exc = KeyError(missing_event_mesg.format(key=key))
        raise new_exc from e


class ExperimentActionBase(Declarative):

    # Name of event that triggers command
    event = d_(Unicode())

    dependencies = List()

    match = Callable()

    # Defines order of invocation. Less than 100 invokes before default. Higher
    # than 100 invokes after default. Note that if concurrent is True, then
    # order of execution is not guaranteed.
    weight = d_(Int(50))

    # Arguments to pass to command by keyword
    kwargs = d_(Dict())

    def _default_dependencies(self):
        return get_dependencies(self.event)

    def _default_match(self):
        code = compile(self.event, 'dynamic', 'eval')
        if len(self.dependencies) == 1:
            return partial(simple_match, self.dependencies[0])
        else:
            return partial(eval, code)


class ExperimentAction(ExperimentActionBase):

    # Command to invoke
    command = d_(Unicode())

    def invoke(self, core, kwargs):
        kwargs = kwargs.copy()
        kwargs.update(self.kwargs)
        core.invoke_command(action.command, parameters=kwargs)


class ExperimentCallback(ExperimentActionBase):

    callback = d_(Callable())
