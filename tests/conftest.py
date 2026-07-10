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
