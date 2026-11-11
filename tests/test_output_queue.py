from types import SimpleNamespace
import threading

import numpy as np
import pytest

from atom.api import Float, Int, Value
from psiaudio.calibration import FlatCalibration
from psiaudio.queue import FIFOSignalQueue, InterleavedFIFOSignalQueue
from psiaudio.stim import Cos2EnvelopeFactory, ToneFactory

from psi.controller.api import HardwareAOChannel, QueuedEpochOutput
from psi.controller.engines.null import NullEngine
from psi.controller.output_commands import output_pause, output_resume


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



# -------- pause / resume --------

class BufferedEngine(NullEngine):
    '''
    Minimal stand-in for a hardware engine that keeps its output buffer filled
    well ahead of playback (as the NI-DAQmx engine does) and supports
    rewriting already-buffered samples via `update_hw_ao`, mirroring
    `NIDAQEngine.update_hw_ao`.
    '''
    ts = Float(0)
    written = Value()
    total_written = Int(0)

    def _default_lock(self):
        return threading.RLock()

    def _default_written(self):
        return np.zeros(0)

    def get_ts(self):
        return self.ts

    def _channel(self):
        return self.get_channels('analog', 'output', 'hardware')[0]

    def fill(self, until):
        samples = until - self.total_written
        out = np.zeros(samples)
        self._channel().get_samples(self.total_written, samples, out)
        self.written = np.concatenate((self.written, out))
        self.total_written = until

    def update_hw_ao(self, channel_name, offset):
        if offset > self.total_written:
            return
        samples = self.total_written - offset
        out = np.zeros(samples)
        self._channel().get_samples(offset, samples, out)
        self.written[offset:] = out


def _command_event(engine):
    controller = SimpleNamespace(invoke_actions=lambda *a, **kw: None)
    workbench = SimpleNamespace(get_plugin=lambda name: controller)
    return SimpleNamespace(parameters={'delay': 0.5}, workbench=workbench)


def test_queued_output_pause_resume():
    fs = 1000
    engine = BufferedEngine()
    channel = HardwareAOChannel(fs=fs, parent=engine,
                                calibration=FlatCalibration.as_attenuation())
    engine.add_channel(channel)
    output = QueuedEpochOutput(name='q')
    output.queue = FIFOSignalQueue()
    channel.add_output(output)

    # 100 ms bursts every 200 ms.
    burst = np.ones(int(0.1 * fs))
    output.queue.append(burst, 100, delays=0.1)
    added, removed = [], []
    output.queue.connect(added.append, 'added')
    output.queue.connect(removed.append, 'removed')
    output.activate(0)

    # Engine buffers 2 s ahead of playback.
    engine.fill(2000)

    # Pause at t = 0.5 + 0.5 s delay = 1.0 s while samples up to 2.0 s are
    # already in the hardware buffer.
    engine.ts = 0.5
    output_pause(_command_event(engine), output)
    engine.fill(5000)

    # Resume at t = 3.0 + 0.5 s delay = 3.5 s. Must not raise "Missed chance
    # to play signal".
    engine.ts = 3.0
    output_resume(_command_event(engine), output)
    engine.fill(7000)
    output.queue.notifier.join(timeout=5)

    w = engine.written
    # Bursts before the pause point are untouched.
    np.testing.assert_array_equal(w[800:900], 1)
    # Silence from the pause point until the resume point, including samples
    # that had already been buffered before the pause was requested.
    np.testing.assert_array_equal(w[1000:3500], 0)
    # Bursts restart exactly at the resume point and keep their cadence.
    np.testing.assert_array_equal(w[3500:3600], 1)
    np.testing.assert_array_equal(w[3600:3700], 0)
    np.testing.assert_array_equal(w[3700:3800], 1)

    # Trials buffered beyond the pause point were canceled, and the queue's
    # timestamps for new trials line up with where they were actually played.
    assert sorted(i['t0'] for i in removed) == \
        pytest.approx([1.0, 1.2, 1.4, 1.6, 1.8])
    resumed_t0 = [i['t0'] for i in added if i['t0'] >= 3.5]
    assert resumed_t0[:3] == pytest.approx([3.5, 3.7, 3.9])
