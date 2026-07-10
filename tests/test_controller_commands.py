"""Unit tests for psi.controller.controller_commands (extracted from
controller/manifest.enaml)."""
from types import SimpleNamespace

from psi.controller.controller_commands import (
    accumulate_actions, start_experiment, stop_experiment,
)


class _FakeController:

    def __init__(self, results):
        self._results = results
        self.wrapup_kwargs = None
        self.start_called = False

    def start_experiment(self):
        self.start_called = True

    def stop_experiment(self, skip_errors, kw):
        # Emulate the real behavior: the first call returns the action
        # results; any subsequent call (state is now 'stopped') returns [].
        results, self._results = self._results, []
        return results

    def _wrapup(self, **kwargs):
        self.wrapup_kwargs = kwargs


def _make_event(controller, parameters=None):
    workbench = SimpleNamespace(get_plugin=lambda name: controller)
    return SimpleNamespace(workbench=workbench, parameters=parameters or {})


def test_start_experiment():
    controller = _FakeController([])
    start_experiment(_make_event(controller))
    assert controller.start_called


def test_stop_experiment_preserves_messages():
    # Regression: this handler used to call controller.stop_experiment a
    # second time to deduplicate results; the second call returned [] (the
    # experiment was already stopped), so the wrapup message was always
    # empty.
    controller = _FakeController(
        ['Saved data to disk', None, 'Saved data to disk ', 'Other note'])
    event = _make_event(controller, {'stop_reason': 'done'})
    stop_experiment(event)
    assert controller.wrapup_kwargs is not None
    message = controller.wrapup_kwargs['message']
    assert 'Saved data to disk' in message
    assert 'Other note' in message
    # Deduplicated: the message appears only once.
    assert message.count('Saved data to disk') == 1


def test_stop_experiment_none_results_skips_wrapup():
    controller = _FakeController([])
    controller.stop_experiment = lambda skip_errors, kw: None
    stop_experiment(_make_event(controller))
    assert controller.wrapup_kwargs is None


def test_accumulate_actions_groups_by_event():
    a1 = SimpleNamespace(event='trial_start', weight=10)
    a2 = SimpleNamespace(event='trial_start', weight=20)
    a3 = SimpleNamespace(event='trial_end', weight=10)
    plugin = SimpleNamespace(_actions=[a1, a2, a3])
    lines = accumulate_actions(plugin)
    assert lines[0] == 'trial_start'
    assert len([line for line in lines if line.startswith('  ')]) == 3
    assert 'trial_end' in lines
