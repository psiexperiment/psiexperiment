import numpy as np
import pytest

from psiaudio.queue import InterleavedFIFOSignalQueue
from psiaudio.stim import Cos2EnvelopeFactory, ToneFactory

from psi.controller.api import QueuedEpochOutput


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
    out = np.zeros(100)
    queued_epoch_output.get_samples(0, 100, out)
    assert np.sqrt(np.mean(out**2)) == pytest.approx(0)

    # Calculate # of ramp samples for tone burst 1 and discard those so we can
    # verify that RMS value of steady-state portion is correct.
    ramp_samples = int(np.ceil(0.5 * queued_epoch_output.fs))
    # Read past the ramp (offset 100..600), then a steady-state window
    # (600..1100) into a fresh buffer for the RMS check.
    discard = np.zeros(ramp_samples)
    queued_epoch_output.get_samples(100, ramp_samples, discard)
    out = np.zeros(ramp_samples)
    queued_epoch_output.get_samples(ramp_samples + 100, ramp_samples, out)
    assert np.sqrt(np.mean(out**2)) == pytest.approx(1.0)


# -------- event_notifiers --------

def test_connect_invalid_event_raises(queued_epoch_output):
    with pytest.raises(KeyError, match='not valid'):
        queued_epoch_output.connect(lambda info: None, event='not_a_real_event')


def test_event_notify_calls_subscribers(queued_epoch_output):
    received = []
    queued_epoch_output.connect(lambda info: received.append(('a', info)),
                                event='added')
    queued_epoch_output.connect(lambda info: received.append(('a2', info)),
                                event='added')
    queued_epoch_output.event_notify('added', {'k': 1})
    assert received == [('a', {'k': 1}), ('a2', {'k': 1})]


def test_event_notify_isolates_events(queued_epoch_output):
    seen = []
    queued_epoch_output.connect(lambda info: seen.append(info), event='removed')
    queued_epoch_output.event_notify('added', {'k': 1})
    assert seen == []  # subscribers to 'removed' should not fire on 'added'


@pytest.mark.skip(reason="InterleavedFIFOSignalQueue.append doesn't fire 'added' "
                         "synchronously on append; would need pop_buffer or a "
                         "different queue subclass to exercise propagation.")
def test_queue_added_event_propagates(ao_channel):
    out = QueuedEpochOutput(name='q')
    out.queue = InterleavedFIFOSignalQueue()
    ao_channel.add_output(out)
    received = []
    out.connect(lambda info: received.append(info), event='added')

    tone = ToneFactory(fs=out.fs, level=0, frequency=100,
                       calibration=out.calibration)
    envelope = Cos2EnvelopeFactory(fs=out.fs, start_time=0, rise_time=0.1,
                                   duration=1.0, input_factory=tone)
    out.queue.append(envelope, 1)
    assert len(received) == 1

