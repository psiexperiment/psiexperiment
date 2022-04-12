import logging
log = logging.getLogger(__name__)

from functools import partial

from atom.api import Str, Int, Dict, Bool, Typed, Callable, List
from enaml.core.api import Declarative, d_

from psi.util import get_dependencies


class ExperimentState(Declarative):
    '''
    Allows for indication of a state (e.g., `experiment_active`, `iti_active`).
    Automatically contributes the start/end events associataed with the state
    (e.g., `experiment_start`, `experiment_end`).
    '''
    name = d_(Str())
    events = ['prepare', 'start', 'end']

    def _generate_events(self):
        events = []
        for name in self.events:
            event_name = '{}_{}'.format(self.name, name)
            event = ExperimentEvent(name=event_name, associated_state=self)
            events.append(event)
        return events


class ExperimentEvent(Declarative):

    name = d_(Str())
    associated_state = Typed(ExperimentState)


missing_event_mesg = '''
Missing event "{key}".

Perhaps an input, output or device is missing from the IO configuration?
'''


def simple_match(key, context, ignore_missing=False):
    try:
        return context[key]
    except Exception as e:
        if ignore_missing:
            return
        new_exc = KeyError(missing_event_mesg.format(key=key))
        raise new_exc from e


def eval_match(code, context, ignore_missing=False):
    try:
        return eval(code, context)
    except Exception as e:
        if ignore_missing:
            return
        raise


class ExperimentActionBase(Declarative):

    # Name of event that triggers command
    event = d_(Str())

    dependencies = List()

    match = Callable()

    # Defines order of invocation. Less than 100 invokes before default. Higher
    # than 100 invokes after default. Note that if concurrent is True, then
    # order of execution is not guaranteed.
    weight = d_(Int(50))

    # Arguments to pass to command by keyword
    kwargs = d_(Dict())

    def _get_params(self, **kwargs):
        kwargs.update(self.kwargs)
        params = {}
        for k, v in kwargs.items():
            if getattr(v, 'is_lookup', False):
                v = v()
            params[k] = v
        return params

    def _default_dependencies(self):
        return get_dependencies(self.event)

    def _default_match(self):
        code = compile(self.event, 'dynamic', 'eval')
        if len(self.dependencies) == 1:
            return partial(simple_match, self.dependencies[0])
        else:
            return partial(eval_match, code)

    def __str__(self):
        return f'{self.event} (weight={self.weight}; kwargs={self.kwargs})'


class ExperimentAction(ExperimentActionBase):

    #: Command to invoke
    command = d_(Str())

    def invoke(self, core, **kwargs):
        params = self._get_params(**kwargs)
        log.debug('Calling command %s with params %r', self.command, params)
        core.invoke_command(self.command, parameters=params)

    def __str__(self):
        return f'ExperimentAction: {self.command}'


class ExperimentCallback(ExperimentActionBase):

    #: Callback to invoke
    callback = d_(Callable())

    def invoke(self, core, **kwargs):
        params = self._get_params(**kwargs)
        self.callback(**params)

    def __str__(self):
        return f'ExperimentCallback: {self.callback}'
