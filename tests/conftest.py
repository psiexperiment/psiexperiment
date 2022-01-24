import pytest

from psiaudio.calibration import FlatCalibration
from psi.controller.api import (EpochOutput, HardwareAOChannel,
                                InterleavedFIFOSignalQueue, QueuedEpochOutput)
from psi.controller.engines.null import NullEngine


@pytest.fixture()
def engine():
    return NullEngine(buffer_size=10)


@pytest.fixture()
def ao_channel(engine):
    channel = HardwareAOChannel(
        fs=1000, calibration=FlatCalibration.as_attenuation(), parent=engine)
    return channel


@pytest.fixture()
def epoch_output(ao_channel):
    output = EpochOutput()
    ao_channel.add_output(output)
    return output


@pytest.fixture()
def queued_epoch_output(ao_channel):
    output = QueuedEpochOutput()
    output.queue = InterleavedFIFOSignalQueue()
    ao_channel.add_output(output)
    return output
