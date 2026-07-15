import pytest

import enaml
with enaml.imports():
    from . import helper_manifest


def test_invalid_experiment_action_eval(workbench, controller):
    workbench.register(helper_manifest.InvalidExperimentActionEval())
    # In 'initialized' state, controller silently ignores missing names.
    # Switch to 'running' so the error surfaces.
    controller.experiment_state = 'running'
    with pytest.raises(NameError):
        controller.invoke_actions('trial_start')


def test_invalid_experiment_action_simple_match(workbench, controller):
    workbench.register(helper_manifest.InvalidExperimentActionSimpleMatch())
    controller.experiment_state = 'running'
    with pytest.raises(KeyError):
        controller.invoke_actions('trial_start')


# -------- startup validation of action dependencies --------

def test_validate_action_dependencies_passes(workbench, controller):
    # The standard helper manifests reference only registered events.
    controller.validate_action_dependencies()


def test_validate_action_dependencies_catches_eval_expression(workbench, controller):
    from psi.core.api import ActionError
    workbench.register(helper_manifest.InvalidExperimentActionEval())
    with pytest.raises(ActionError, match='no_such_event'):
        controller.validate_action_dependencies()


def test_validate_action_dependencies_suggests_close_match(workbench, controller):
    from psi.core.api import ActionError
    controller.register_action('experiment_endz', 'store_result')
    with pytest.raises(ActionError) as excinfo:
        controller.validate_action_dependencies()
    mesg = str(excinfo.value)
    assert 'experiment_endz' in mesg
    # The suggestion names the correctly-spelled event.
    assert 'Did you mean' in mesg
    assert 'experiment_end' in mesg


def test_validate_action_dependencies_allows_builtins(workbench, controller):
    # eval-based expressions may use builtins; these must not be flagged.
    controller.register_action('bool(experiment_start)', 'store_result')
    controller.validate_action_dependencies()


def test_validate_action_dependencies_opt_out(workbench, controller):
    from psi.core.api import ExperimentAction
    action = ExperimentAction(event='dynamically_generated_event',
                              command='store_result',
                              allow_unregistered=True)
    controller._registered_actions.append(action)
    controller.validate_action_dependencies()


def test_start_experiment_runs_validation(workbench, controller, monkeypatch):
    # _start_experiment must validate before invoking any actions. Atom
    # instances are slot-based, so patch at the class level.
    cls = type(controller)
    invoked = []
    monkeypatch.setattr(cls, 'invoke_actions',
                        lambda self, *a, **kw: invoked.append(a))

    def fail(self):
        raise RuntimeError('validation sentinel')
    monkeypatch.setattr(cls, 'validate_action_dependencies', fail)

    with pytest.raises(RuntimeError, match='validation sentinel'):
        controller._start_experiment()
    assert invoked == []
