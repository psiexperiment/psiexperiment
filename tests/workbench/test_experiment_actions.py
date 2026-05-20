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
