import pytest

import enaml
with enaml.imports():
    from . import helper_manifest


def test_invalid_experiment_action_eval(workbench, controller):
    workbench.register(helper_manifest.InvalidExperimentActionEval())
    with pytest.raises(NameError):
        controller.invoke_actions('trial_start')


def test_invalid_experiment_action_simple_match(workbench, controller):
    workbench.register(helper_manifest.InvalidExperimentActionSimpleMatch())
    with pytest.raises(KeyError):
        controller.invoke_actions('trial_start')
