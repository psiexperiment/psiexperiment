'''
The paradigm name travels on the workbench.

`get_default_path` used to read it from a process global in psi.runtime,
even though `PSIWorkbench.start_workspace` was already handed it and
dropped it on the floor. These tests are here rather than beside the
other command tests because they use the real workbench: a stub carrying
the attribute cannot notice if the attribute stops existing.
'''
from pathlib import Path

import pytest

from psi.experiment.experiment_commands import get_default_path


def test_workbench_carries_it(workbench):
    workbench.experiment_name = 'abr_io'
    assert workbench.experiment_name == 'abr_io'


def test_default_path_reads_it(workbench, tmp_path, monkeypatch):
    from psi import config as psi_config

    monkeypatch.setenv('PSI_BASE_DIRECTORY', str(tmp_path / 'base'))
    psi_config.reload_config()

    workbench.experiment_name = 'abr_io'
    path = Path(get_default_path(workbench, 'preferences'))
    assert path.name == 'abr_io'
    assert path.parent.name == 'preferences'


def test_unset_is_reported(workbench):
    '''
    A fresh workbench has not been through start_workspace, which is the
    step that would have set it.
    '''
    with pytest.raises(ValueError, match='experiment name'):
        get_default_path(workbench, 'layout')
