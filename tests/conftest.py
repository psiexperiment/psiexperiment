import pytest

from psi.controller.calibration import FlatCalibration
from psi.controller.channel import HardwareAOChannel
from psi.controller.output import EpochOutput, QueuedEpochOutput
from psi.controller.queue import InterleavedFIFOSignalQueue
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
