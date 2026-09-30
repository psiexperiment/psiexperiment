'''
Tests for :mod:`psi.launcher.experiment`.
'''
import json

from atom.api import Enum, Str

from psi.experiment.api import ParadigmNotFound
from psi.launcher import experiment as experiment_module
from psi.launcher.experiment import Experiment, FrozenExperiment, load_experiments

from launcher_helpers import FakeParadigm


class LabExperiment(Experiment):
    '''
    A lab's own row type, as a launcher would define it: one extra setting
    that must survive saving and freezing.
    '''
    rig = Str()


class LabFrozenExperiment(FrozenExperiment):
    rig = Str()
    ear = Enum('selected', 'left', 'right')


LabExperiment.frozen_class = LabFrozenExperiment


def test_base_frozen_experiment_has_no_ear():
    # Ear is a CFTS concept and lives in cfts's subclass.
    assert 'ear' not in FrozenExperiment.members()


def test_freeze_defaults_to_first_mode():
    exp = Experiment(FakeParadigm(modes=['ipsi', 'contra']))
    assert exp.freeze().mode == 'ipsi'


def test_freeze_without_modes_uses_run():
    exp = Experiment(FakeParadigm())
    assert exp.modes == ['Run']
    assert exp.freeze().mode == 'Run'


def test_freeze_copies_row():
    paradigm = FakeParadigm(plugins=['temperature'], preferences=['a/default.preferences'])
    exp = Experiment(paradigm, plugins=['temperature'], preference='default.preferences',
                     mode_notes={'run': 'note'})
    frozen = exp.freeze('run')
    assert type(frozen) is FrozenExperiment
    assert frozen.paradigm is paradigm
    assert frozen.plugins == ['temperature']
    assert frozen.preference == exp.preference
    assert frozen.mode_notes == {'run': 'note'}


def test_freeze_uses_frozen_class_and_carries_subclass_state():
    exp = LabExperiment(FakeParadigm(), rig='booth-2')
    frozen = exp.freeze('run', ear='left')
    assert type(frozen) is LabFrozenExperiment
    assert frozen.rig == 'booth-2'
    assert frozen.ear == 'left'


def test_refreeze_keeps_class_and_mode():
    frozen = LabExperiment(FakeParadigm(modes=['a', 'b'])).freeze('b', ear='right')
    again = frozen.freeze(ear='left')
    assert type(again) is LabFrozenExperiment
    assert again.mode == 'b'
    assert again.ear == 'left'


def test_state_is_json_serializable():
    exp = Experiment(FakeParadigm(preferences=['x/default.preferences']),
                     preference='default.preferences')
    state = json.loads(json.dumps(exp.__getstate__()))
    assert state['paradigm'] == 'tone'
    assert state['preference'] == 'default.preferences'
    assert 'preferences' not in state


def test_empty_preference_is_not_warned_about(caplog):
    Experiment(FakeParadigm(), preference='')
    assert 'Invalid preference' not in caplog.text


def test_invalid_plugins_are_dropped():
    exp = Experiment(FakeParadigm(plugins=['a']), plugins=['a', 'renamed'])
    assert exp.plugins == ['a']


def test_load_experiments_round_trip(monkeypatch):
    paradigms = {'tone': FakeParadigm('tone'), 'noise': FakeParadigm('noise')}

    def get_paradigm(name):
        try:
            return paradigms[name]
        except KeyError:
            raise ParadigmNotFound(name) from None

    monkeypatch.setattr(experiment_module.paradigm_manager, 'get_paradigm', get_paradigm)

    saved = [
        LabFrozenExperiment(paradigms['tone'], mode='Run', rig='r1', ear='left').__getstate__(),
        {'paradigm': 'removed', 'mode': 'Run'},
        # Legacy keys written by older launchers are ignored.
        {'paradigm': 'noise', 'mode': 'Run', 'label': 'old', 'modes': ['Run'],
         'preferences': [], 'mode_notes': {'RUN': 'loud'}},
    ]
    loaded = load_experiments(json.loads(json.dumps(saved)), LabFrozenExperiment)
    assert [e.paradigm.name for e in loaded] == ['tone', 'noise']
    assert loaded[0].ear == 'left'
    assert loaded[0].rig == 'r1'
    assert loaded[1].mode_notes == {'run': 'loud'}
