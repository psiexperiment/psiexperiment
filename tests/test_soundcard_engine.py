'''
Tests for `SoundcardEngine` that need no sound card: the stream it opens
(`PlayRec`) is replaced by a stand-in where needed.
'''
from types import SimpleNamespace

import numpy as np
import pytest

import psi.controller.engines.soundcard as soundcard_module
from psi.controller.engines.soundcard.playrec import XRUN_FLAGS
from psi.controller.engines.soundcard import (
    SoundcardEngine, SoundcardHardwareAOChannel,
)


def test_clone_keeps_settings():
    # The in-ear calibration runs on clones (`psi.controller.calibration.
    # acquire`). A clone left at the default 96 kHz failed to open an output
    # running at 44.1 kHz.
    engine = SoundcardEngine(name='sound', fs=44100, device_name='device',
                             blocksize=512, latency_samples=10)
    engine.add_channel(
        SoundcardHardwareAOChannel(name='speaker_1', channel=0))
    clone = engine.clone()
    assert (clone.fs, clone.blocksize, clone.latency_samples) == \
        (44100, 512, 10)
    assert clone.device_name == 'device'
    assert [c.name for c in clone.get_channels(active=False)] == \
        ['speaker_1']


class FakeTimingStream:
    '''
    A started stream, as far as `SoundcardEngine`'s clock sees it.
    '''

    def __init__(self, time):
        self.stream = SimpleNamespace(time=time)
        self.callback = SimpleNamespace(
            status_counts=dict.fromkeys(XRUN_FLAGS, 0),
            output_lag_floor=np.nan,
            take_callback_stats=lambda: (0.0156, 0.0002))


def running_engine(t0=100):
    engine = SoundcardEngine(name='sound', fs=48000, device_name='device',
                             blocksize=512)
    engine._stream = FakeTimingStream(t0)
    engine.t0 = t0
    engine._running = True
    return engine


def test_output_lag():
    engine = running_engine()
    assert np.isnan(engine.output_lag())
    # Blocks reach the DAC 20 ms after their offsets say (output latency).
    engine._stream.callback.output_lag_floor = 100.02
    assert engine.output_lag() == pytest.approx(0.02)


def test_clock_follows_samples_played():
    # The keypress-to-sound lag that grew over a session: each gap the
    # device filled with silence delayed every later stimulus, since
    # stimuli are placed at round(get_ts() * fs). get_ts now subtracts the
    # growth of the output lag, eased in so time never runs backwards.
    engine = running_engine()
    stream, callback = engine._stream.stream, engine._stream.callback
    stream.time = 101
    callback.output_lag_floor = 100.02
    assert engine.get_ts() == pytest.approx(1)

    # 50 ms of silence.
    callback.output_lag_floor = 100.07
    times = []
    for stream.time in [101.1, 101.2, 102]:
        times.append(engine.get_ts())
    # At most 0.25 s per s (CLOCK_SLEW) until the 50 ms is made up.
    assert times == pytest.approx([1.075, 1.15, 1.95])

    # Never undone: silence only adds to the lag.
    callback.output_lag_floor = 100.04
    stream.time = 103
    assert engine.get_ts() == pytest.approx(2.95)


def test_clock_without_dac_times():
    # Host APIs that report no DAC times keep the plain stream clock.
    engine = running_engine()
    engine._stream.stream.time = 160
    assert engine.get_ts() == pytest.approx(60)


def test_log_timing_reports_lag_growth_and_new_xruns(caplog):
    engine = running_engine()
    stream, callback = engine._stream.stream, engine._stream.callback
    caplog.set_level('INFO', logger=soundcard_module.log.name)

    stream.time = 101
    callback.output_lag_floor = 100.02
    engine.log_timing()
    record, = caplog.records
    assert record.levelname == 'INFO'
    assert 'at 1.0 s: output lag 20.0 ms (+0.0 ms since start, clock ' \
        'corrected by 0.0 ms); xruns: none' in record.getMessage()

    # The device played 30 ms of silence, and the clock has caught up.
    caplog.clear()
    callback.status_counts['output_underflow'] = 3
    callback.output_lag_floor = 100.05
    stream.time = 160
    engine.log_timing()
    record, = caplog.records
    assert record.levelname == 'WARNING'
    assert "output lag 50.0 ms (+30.0 ms since start, clock corrected by " \
        "30.0 ms); xruns: output_underflow 3 ({'output_underflow': 3} new)" \
        in record.getMessage()

    # No new xruns since: back to INFO, with the running totals.
    caplog.clear()
    engine.log_timing()
    record, = caplog.records
    assert record.levelname == 'INFO'
    assert 'xruns: output_underflow 3' in record.getMessage()
    assert 'new' not in record.getMessage()
    assert 'longest callback gap 15.6 ms, longest callback 0.2 ms ' \
        '(block is 10.7 ms)' in record.getMessage()


class FakeAudioTime:
    '''
    Stands in for the `time` module in `psi.controller.engine`: each wait
    advances the clock, and the audio thread delivers `block` more input
    samples.
    '''

    def __init__(self, engine, block):
        self.engine = engine
        self.block = block
        self.now = 0.0
        self.sleeps = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.sleeps += 1
        self.engine._total_samples_read += self.block


def recording_engine(read, has_input=True):
    '''
    A running engine 1 s (48000 samples) in that has read `read` samples
    of input.
    '''
    engine = running_engine()
    engine._stream.stream.time = 101
    engine._has_input = has_input
    engine._total_samples_read = read
    return engine


def test_input_time():
    engine = recording_engine(24000)
    assert engine.input_time() == pytest.approx(0.5)
    engine.latency_samples = 100
    assert engine.input_wait_timeout() == pytest.approx((100 + 2 * 512)
                                                        / 48000)


def test_input_time_without_input():
    '''
    An output-only stream has nothing to wait for.
    '''
    assert recording_engine(0, has_input=False).input_time() is None


def test_stop_waits_for_input(monkeypatch):
    '''
    Stopping as an output ended used to cut off the end of the recording of
    it (the latency plus up to a block). `wait_for_inputs`, which the IO
    manager calls before stopping the engines, keeps the stream open until
    the input has read up to the engine's clock.
    '''
    from psi.controller import engine as engine_module
    engine = recording_engine(47000)
    clock = FakeAudioTime(engine, 512)
    monkeypatch.setattr(engine_module, 'time', clock)
    assert engine_module.wait_for_inputs([engine]) == []
    assert engine._total_samples_read >= 48000
    assert clock.sleeps == 2
