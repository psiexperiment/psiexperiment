import pytest

from psiaudio.queue import InterleavedFIFOSignalQueue
from psiaudio.calibration import FlatCalibration

# Note: the package layering is enforced by import-linter and
# tools/check_enaml_layering.py, so import order no longer matters here.
# (Historically this import doubled as a priming step to pre-resolve
# circular imports between the plugin packages.)
from psi.controller.api import (EpochOutput, HardwareAIChannel,
                                HardwareAOChannel, QueuedEpochOutput)
from psi.controller.engines.null import NullEngine


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    '''
    Give every test its own empty configuration file.

    Settings resolve as default, then config.toml, then the environment,
    and the file is whatever PSI_CONFIG_FILE points at -- on a developer
    machine or a rig, the real one. Without this a test asserting a
    built-in default passes or fails depending on what the person running
    it has configured, and anything that saves a setting writes into
    their live file. The tests that exercise the configuration system
    itself set PSI_CONFIG_FILE again for their own purposes, which
    overrides this.
    '''
    from psi import config as psi_config

    monkeypatch.setenv('PSI_CONFIG_FILE', str(tmp_path / 'config.toml'))
    psi_config.reload_config()
    yield
    psi_config.reload_config()


@pytest.fixture(scope='session')
def app():
    """
    The enaml application.

    Enaml allows only one instance per process, so every test that needs
    one shares this. Qt is imported lazily so a run that does not touch
    the GUI does not pull it in.
    """
    from enaml.application import Application
    from enaml.qt.qt_application import QtApplication
    return Application.instance() or QtApplication()


@pytest.fixture()
def engine():
    return NullEngine(buffer_size=10)


@pytest.fixture()
def ao_channel(engine):
    channel = HardwareAOChannel(
        fs=1000,
        calibration=FlatCalibration.as_attenuation(),
        parent=engine,
    )
    # Engine.initialized() only wires `channel.engine` for children present at
    # construction time; channels attached afterwards must be registered.
    engine.add_channel(channel)
    return channel


@pytest.fixture()
def ai_channel(engine):
    channel = HardwareAIChannel(
        name='ai',
        fs=100e3,
        calibration=FlatCalibration.as_attenuation(),
        parent=engine,
    )
    engine.add_channel(channel)
    return channel


@pytest.fixture()
def epoch_output(ao_channel):
    output = EpochOutput(name='test')
    ao_channel.add_output(output)
    return output


@pytest.fixture()
def queued_epoch_output(ao_channel):
    output = QueuedEpochOutput()
    output.queue = InterleavedFIFOSignalQueue()
    ao_channel.add_output(output)
    return output
