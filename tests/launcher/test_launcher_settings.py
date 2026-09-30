'''
Tests for :mod:`psi.launcher.settings`.
'''
from pathlib import Path

import pytest

from psi.launcher.experiment import Experiment
from psi.launcher.settings import LauncherSettings

from launcher_helpers import FakeParadigm, FakeProcessManager


class Settings(LauncherSettings):

    def build_experiment(self, experiment, save=True):
        if experiment.paradigm.name == 'broken':
            raise ValueError('not configured')
        path = self.data_path(self.animal, experiment.mode) if save else None
        return self.psi_command(experiment, path), {'MODE': experiment.mode}


@pytest.fixture
def settings():
    return Settings(process_manager=FakeProcessManager(), animal='m1')


def test_build_experiment_must_be_implemented():
    settings = LauncherSettings(process_manager=FakeProcessManager())
    with pytest.raises(NotImplementedError):
        settings.build_experiment(Experiment(FakeParadigm()).freeze())


def test_run_experiment_queues_and_opens(settings):
    exp = Experiment(FakeParadigm(modes=['ipsi', 'contra']))
    settings.run_experiment(exp, 'contra', autostart=True)
    pm = settings.process_manager
    [(cmd, env, uid)] = pm.commands
    assert env == {'MODE': 'contra'}
    assert cmd[:2] == ['psi', 'lab.paradigms.tone']
    assert pm.autostart
    assert pm.opened == 1


def test_run_discards_leftover_commands(settings):
    pm = settings.process_manager
    pm.add_command(['stale'], {})
    settings.run_experiment(Experiment(FakeParadigm()), 'Run')
    assert [c[0][0] for c in pm.commands] == ['psi']


def test_failed_build_queues_nothing(settings):
    pm = settings.process_manager
    sequence = [Experiment(FakeParadigm()).freeze(),
                Experiment(FakeParadigm('broken')).freeze()]
    with pytest.raises(ValueError):
        settings.run_sequence(sequence)
    assert pm.commands == []
    assert pm.opened == 0


def test_psi_command(settings):
    paradigm = FakeParadigm(plugins=['temperature'], preferences=['p/default.preferences'])
    exp = Experiment(paradigm, plugins=['temperature'], preference='default.preferences')
    settings.logging_level = 'debug'
    cmd = settings.psi_command(exp, Path('out'))
    assert cmd == [
        'psi', 'lab.paradigms.tone', 'out',
        '--preferences', str(Path('p/default.preferences')),
        '--debug-level-console', 'DEBUG',
        '--plugin', 'temperature',
    ]


def test_psi_command_skips_empty_preference(settings):
    cmd = settings.psi_command(Experiment(FakeParadigm()))
    assert '--preferences' not in cmd


def test_data_path_drops_empty_parts(settings, monkeypatch):
    monkeypatch.setenv('PSI_DATA_ROOT', str(Path('data')))
    from psi import config
    config.reload_config()
    path = settings.data_path('m1', '', 'note  with  spaces', None, 'abr')
    assert path == Path('data') / '{date_time} m1 note with spaces abr'
