import pytest


from psi.controller.experiment_action import (
    EventLogger, ExperimentAction, ExperimentCallback, ExperimentEvent,
    ExperimentState, eval_match, simple_match,
)


@pytest.fixture
def simple_experiment_action():
    return ExperimentAction(event='experiment_end')


@pytest.fixture
def complex_experiment_action():
    return ExperimentAction(event='experiment_end and not task_active')


@pytest.fixture
def context():
    return {
        'experiment_end': True,
        'task_active': False,
    }


def test_evaluate_simple(simple_experiment_action, context):
    assert simple_experiment_action.match(context) == True


def test_evaluate_complex(complex_experiment_action, context):
    assert complex_experiment_action.match(context) == True


def test_evaluate(complex_experiment_action):
    context = {'experiment_end': False, 'task_active': False}
    result = complex_experiment_action.match(context)
    assert result == False

    context = {'experiment_end': True, 'task_active': True}
    result = complex_experiment_action.match(context)
    assert result == False

    context = {'experiment_end': False, 'task_active': True}
    result = complex_experiment_action.match(context)
    assert result == False


# -------- match helpers --------

def test_simple_match_missing_key_raises():
    with pytest.raises(KeyError, match='Missing event'):
        simple_match('nope', {})


def test_simple_match_missing_key_ignored():
    assert simple_match('nope', {}, ignore_missing=True) is None


def test_eval_match_missing_name_raises():
    code = compile('a and b', 'dynamic', 'eval')
    with pytest.raises(NameError):
        eval_match(code, {})


def test_eval_match_missing_name_ignored():
    code = compile('a and b', 'dynamic', 'eval')
    assert eval_match(code, {}, ignore_missing=True) is None


# -------- dependency detection picks the right matcher --------

def test_simple_event_uses_simple_match():
    action = ExperimentAction(event='trial_start')
    assert action.dependencies == ['trial_start']
    # functools.partial wraps simple_match for single-dep actions.
    assert action.match.func is simple_match


def test_compound_event_uses_eval_match():
    action = ExperimentAction(event='a and not b')
    assert set(action.dependencies) == {'a', 'b'}
    assert action.match.func is eval_match


# -------- weight, kwargs, invocations --------

def test_action_default_weight_is_50():
    action = ExperimentAction(event='x', command='do')
    assert action.weight == 50


def test_actions_sort_by_weight():
    a = ExperimentAction(event='x', command='one', weight=10)
    b = ExperimentAction(event='x', command='two', weight=200)
    c = ExperimentAction(event='x', command='three', weight=50)
    actions = sorted([b, a, c], key=lambda x: x.weight)
    assert [act.command for act in actions] == ['one', 'three', 'two']


def test_invoke_passes_kwargs_and_increments_counter():
    captured = {}

    class FakeCore:
        def invoke_command(self, command, parameters):
            captured['command'] = command
            captured['parameters'] = parameters

    action = ExperimentAction(event='x', command='do_thing',
                              kwargs={'reason': 'go'})
    action.invoke(FakeCore(), timestamp=1.23, event='x', extra=99)
    assert action.invocations == 1
    assert captured['command'] == 'do_thing'
    # `kwargs` overrides any same-named caller kwargs (see _get_params).
    assert captured['parameters']['reason'] == 'go'
    assert captured['parameters']['extra'] == 99


def test_invoke_delay_uses_timed_call(monkeypatch):
    calls = []

    def fake_timed_call(delay_ms, fn, *args, **kwargs):
        calls.append((delay_ms, fn, args, kwargs))

    monkeypatch.setattr(
        'psi.controller.experiment_action.timed_call',
        fake_timed_call,
    )

    class FakeCore:
        def invoke_command(self, *args, **kwargs):
            calls.append(('invoked', args, kwargs))

    action = ExperimentAction(event='x', command='go', delay=0.25)
    action.invoke(FakeCore())
    # The command is NOT invoked synchronously; it's scheduled.
    assert calls and calls[0][0] == 250  # ms
    assert all(c[0] != 'invoked' for c in calls)


# -------- ExperimentCallback --------

def test_callback_invokes_callable():
    received = {}

    def cb(**kwargs):
        received.update(kwargs)
        return 'ok'

    action = ExperimentCallback(event='x', callback=cb, kwargs={'a': 1})
    result = action.invoke(core=None, b=2)
    assert result == 'ok'
    assert received == {'a': 1, 'b': 2}


# -------- ExperimentState --------

def test_experiment_state_generates_three_events():
    state = ExperimentState(name='trial')
    events = state._generate_events()
    names = [e.name for e in events]
    assert names == ['trial_prepare', 'trial_start', 'trial_end']
    for e in events:
        assert isinstance(e, ExperimentEvent)
        assert e.associated_state is state


# -------- EventLogger --------

def test_event_logger_invokes_command_with_data():
    captured = {}

    class FakeCore:
        def invoke_command(self, command, parameters):
            captured['command'] = command
            captured['parameters'] = parameters

    logger = EventLogger(command='log.write')
    logger._invoke(FakeCore(), {'event': 'trial_start', 'timestamp': 1.0})
    assert captured['command'] == 'log.write'
    assert captured['parameters']['data']['event'] == 'trial_start'
