import numpy as np
import pytest

from psiaudio.stim import Cos2EnvelopeFactory, ToneFactory


@pytest.fixture()
def tb1(queued_epoch_output):
    tone = ToneFactory(fs=queued_epoch_output.fs, level=0, frequency=100,
                       calibration=queued_epoch_output.calibration)
    envelope = Cos2EnvelopeFactory(fs=queued_epoch_output.fs, start_time=0,
                                   rise_time=0.5, duration=5,
                                   input_factory=tone)
    return envelope


@pytest.fixture()
def tb2(queued_epoch_output):
    tone = ToneFactory(fs=queued_epoch_output.fs, level=0, frequency=250,
                       calibration=queued_epoch_output.calibration)
    envelope = Cos2EnvelopeFactory(fs=queued_epoch_output.fs, start_time=0,
                                   rise_time=0.25, duration=5,
                                   input_factory=tone)
    return envelope


def test_queue_delay(queued_epoch_output, tb1, tb2):
    queued_epoch_output.queue.append(tb1, 2)
    queued_epoch_output.queue.append(tb2, 2)
    queued_epoch_output.activate(100)

    # Ensure that we get a strin gof zeros.
    out = np.empty(100)
    queued_epoch_output.get_samples(0, 100, out)
    assert np.sqrt(np.mean(out**2)) == pytest.approx(0)

    # Calculate # of ramp samples for tone bust 1 and discard those so we can
    # verify that RMS value of steady-state portion is correct.
    ramp_samples = int(np.ceil(0.5 * queued_epoch_output.fs))
    out = np.empty(ramp_samples)
    queued_epoch_output.get_samples(100, ramp_samples, out)
    queued_epoch_output.get_samples(ramp_samples + 100, ramp_samples, out)
    assert np.sqrt(np.mean(out**2)) == pytest.approx(1.0)
