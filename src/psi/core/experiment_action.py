import logging
log = logging.getLogger(__name__)

from functools import partial

from atom.api import Bool, Str, Int, Dict, Float, Typed, Callable, List
from enaml.application import timed_call
from enaml.core.api import Declarative, d_

from psi.util import get_dependencies


################################################################################
# Action weights
#
# Actions bound to the same event run in order of weight, lowest first (ties
# in the order they were registered). These name the actions others are
# ordered against, so an action that must run before or after one says so
# (e.g., `weight = STOP_ENGINES_WEIGHT - 1` to run while the engines are
# still running). To run after something has happened, binding to the event
# it raises (e.g., `engines_stopped`) is clearer still.
################################################################################

#: Weight of an action that doesn't set one.
DEFAULT_WEIGHT = 50

#: `experiment_prepare`: start the calibration
#: (`psi.controller.calibration.start`).
CALIBRATION_START_WEIGHT = 10

#: `experiment_prepare`: configure the engines
#: (`psi.controller.configure_engines`).
CONFIGURE_ENGINES_WEIGHT = 100

#: `experiment_start`: start the engines (`psi.controller.start_engines`).
START_ENGINES_WEIGHT = DEFAULT_WEIGHT

#: `experiment_end`: stop the engines (`psi.controller.stop_engines`). First,
#: so the streams stop before anything else runs: work done while the
#: engines wind down (e.g., restyling the window, redrawing the plots) can
#: hold the GIL long enough to make a sound card callback miss its deadline,
#: which plays silence. Anything after this also sees all the data.
STOP_ENGINES_WEIGHT = 0

#: `experiment_end`: flush the stores, after everything else has added to
#: them.
FLUSH_STORES_WEIGHT = 1000


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
    except Exception:
        if ignore_missing:
            return
        raise


class ExperimentActionBase(Declarative):

    # Name of event that triggers command
    event = d_(Str())

    dependencies = List()

    match = Callable()

    #: Order among the actions bound to the same event: lowest first (see
    #: the weights at the top of this module). Note that if concurrent is
    #: True, then order of execution is not guaranteed.
    weight = d_(Int(DEFAULT_WEIGHT))

    #: Arguments to pass to command by keyword
    kwargs = d_(Dict())

    #: Should action be delayed? If nonzero, this may cause some timing issues.
    #: Use with caution.
    delay = d_(Float(0))

    #: If True, the controller's startup validation will not require the
    #: names referenced by `event` to be registered events or state flags.
    #: Only set this for events that are generated dynamically at runtime.
    allow_unregistered = d_(Bool(False))

    #: Number of times the command was invoked 
    invocations = Int(0)

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

    def invoke(self, core, **kwargs):
        self.invocations += 1
        log.debug(f'Action {self} invoked {self.invocations} times')
        if self.delay != 0:
            log.debug(f'Queueing {self} to be invoked in {self.delay} seconds')
            timed_call(self.delay * 1e3, self._invoke, core, **kwargs)
        else:
            return self._invoke(core, **kwargs)


class ExperimentAction(ExperimentActionBase):

    #: Command to invoke
    command = d_(Str())

    def _invoke(self, core, **kwargs):
        params = self._get_params(**kwargs)
        log.debug('Calling command %s with params %r', self.command, params)
        return core.invoke_command(self.command, parameters=params)

    def __str__(self):
        return f'On {self.event} call {self.command}'

    def __repr__(self):
        return f'<ExperimentAction: {self} {self.kwargs}>'


class ExperimentCallback(ExperimentActionBase):

    #: Callback to invoke
    callback = d_(Callable())

    def _invoke(self, core, **kwargs):
        params = self._get_params(**kwargs)
        return self.callback(**params)

    def __str__(self):
        return f'ExperimentCallback: {self.callback}'


class EventLogger(Declarative):
    '''
    Defines a command that is invoked whenever an event occurs
    '''
    command = d_(Str())

    def _invoke(self, core, kwargs):
        return core.invoke_command(self.command, parameters={'data': kwargs})

    def __str__(self):
        return f'EventLogger: {self.callback}'
