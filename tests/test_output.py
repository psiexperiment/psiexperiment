import pytest

import numpy as np

from psiaudio.stim import Cos2EnvelopeFactory, ToneFactory

from psi.controller.api import (
    MUXOutput, NullOutput, RampedEpochOutput, Synchronized,
)
from psi.controller.engines.null import NullEngine


@pytest.fixture()
def tb1(epoch_output):
    tone = ToneFactory(fs=epoch_output.fs, level=0, frequency=100,
                       calibration=epoch_output.calibration)
    envelope = Cos2EnvelopeFactory(fs=epoch_output.fs, start_time=0,
                                   rise_time=0.5, duration=5,
                                   input_factory=tone)
    return envelope


@pytest.fixture()
def tb2(epoch_output):
    tone = ToneFactory(fs=epoch_output.fs, level=0, frequency=250,
                       calibration=epoch_output.calibration)
    envelope = Cos2EnvelopeFactory(fs=epoch_output.fs, start_time=0,
                                   rise_time=0.25, duration=5,
                                   input_factory=tone)
    return envelope


def test_epoch_output_pause(epoch_output, tb1):
    full_waveform1 = tb1.get_samples_remaining()
    tb1.reset()

    epoch_output.source = tb1
    epoch_output.activate(0)

    out = np.zeros(1000)
    epoch_output.get_samples(0, 1000, out)
    np.testing.assert_array_equal(full_waveform1[:1000], out)

    # Pause on EpochOutput is an immediate boolean gate (the timestamp is
    # ignored): while paused, get_samples leaves the buffer untouched and does
    # not advance the playback position.
    epoch_output.pause(1000 / epoch_output.fs)
    out = np.zeros(1000)
    epoch_output.get_samples(1000, 1000, out)
    assert np.all(out == 0)

    # After resume, playback continues from where it left off.
    epoch_output.resume(2000 / epoch_output.fs)
    out = np.zeros(1000)
    epoch_output.get_samples(1000, 1000, out)
    np.testing.assert_array_equal(full_waveform1[1000:2000], out)


def test_epoch_output_buffer(epoch_output, tb1):
    """Verify sequential reads advance _offset and produce contiguous output."""
    full_waveform1 = tb1.get_samples_remaining()
    tb1.reset()

    epoch_output.source = tb1
    epoch_output.activate(0)

    # First block.
    out = np.zeros(1000, dtype=np.double)
    epoch_output.get_samples(0, 1000, out)
    np.testing.assert_array_almost_equal(out, full_waveform1[:1000])

    # Second block — _offset is now 1000 so requesting offset=1000 reads the
    # next contiguous chunk.
    out = np.zeros(1000, dtype=np.double)
    epoch_output.get_samples(1000, 1000, out)
    np.testing.assert_array_almost_equal(out, full_waveform1[1000:2000])

    # Requesting samples starting after the most recently delivered offset
    # must raise (offset > _offset = "missed chance" in get_samples).
    out = np.zeros(1000, dtype=np.double)
    with pytest.raises(ValueError, match='Missed chance'):
        epoch_output.get_samples(3000, 1000, out)


# -------- NullOutput --------

def test_null_output_writes_zeros(ao_channel):
    out = NullOutput()
    ao_channel.add_output(out)
    out.activate(0)
    buf = np.full(100, 5.0)
    out.get_samples(0, 100, buf)
    # NullOutput adds zeros to buf — value is unchanged.
    np.testing.assert_array_equal(buf, np.full(100, 5.0))


# -------- MUXOutput --------

def test_mux_output_add_remove(ao_channel):
    mux = MUXOutput()
    ao_channel.add_output(mux)
    a = NullOutput()
    b = NullOutput()
    mux.add_output(a)
    mux.add_output(b)
    assert a in mux.outputs and b in mux.outputs
    # Adding the same output twice is idempotent.
    mux.add_output(a)
    assert mux.outputs.count(a) == 1
    mux.remove_output(a)
    assert a not in mux.outputs
    # Removing something not present is a no-op.
    mux.remove_output(a)


def test_mux_output_sums_children(ao_channel):
    mux = MUXOutput()
    ao_channel.add_output(mux)
    a = NullOutput()
    b = NullOutput()
    mux.add_output(a)
    mux.add_output(b)
    a.activate(0)
    b.activate(0)
    buf = np.full(50, 3.0)
    mux.get_samples(0, 50, buf)
    # Two NullOutputs each add 0; buf is unchanged.
    np.testing.assert_array_equal(buf, np.full(50, 3.0))


# -------- Synchronized --------

def test_synchronized_outputs_is_children():
    sync = Synchronized()
    a = NullOutput(parent=sync)
    b = NullOutput(parent=sync)
    assert sync.outputs == [a, b]


def test_synchronized_engines_collects_unique_engines(ao_channel, engine):
    # `engine` is a read-only Property on Output (computed from .channel).
    # Wire up real channels so each output's engine resolves correctly.
    cal = ao_channel.calibration
    from psi.controller.api import HardwareAOChannel
    second_engine = NullEngine(buffer_size=10)
    other_channel = HardwareAOChannel(name='speaker_b', fs=1000,
                                      calibration=cal)
    second_engine.add_channel(other_channel)

    sync = Synchronized()
    a = NullOutput(parent=sync)
    b = NullOutput(parent=sync)
    ao_channel.add_output(a)
    other_channel.add_output(b)

    assert sync.engines == {engine, second_engine}


# -------- EpochOutput edge cases --------

def test_epoch_output_missed_chance_raises(epoch_output):
    tone = ToneFactory(fs=epoch_output.fs, level=0, frequency=100,
                       calibration=epoch_output.calibration)
    epoch_output.source = tone
    epoch_output.activate(0)
    out = np.empty(100)
    epoch_output.get_samples(0, 100, out)
    # Now _offset is 100. Asking for offset 200 (gap of 100 samples) means
    # we missed our chance to play 100..199.
    with pytest.raises(ValueError, match='Missed chance'):
        epoch_output.get_samples(200, 100, out)


def test_epoch_output_inactive_returns_unchanged(epoch_output):
    # Without activation, EpochOutput.get_samples short-circuits (active=False).
    out = np.full(50, 7.0)
    epoch_output.get_samples(0, 50, out)
    np.testing.assert_array_equal(out, np.full(50, 7.0))


def test_epoch_output_rejects_downstream_outputs(epoch_output):
    # Regression: this error message used to reference an undefined variable,
    # raising NameError instead of ValueError naming the output.
    with pytest.raises(ValueError, match='Output test does not accept'):
        epoch_output.add_output(NullOutput())


# -------- RampedEpochOutput --------

def test_ramped_epoch_output_creates_ramp(ao_channel):
    ramped = RampedEpochOutput(name='ramped')
    ao_channel.add_output(ramped)
    waveform = np.ones(1000)
    ramped.set_waveform(waveform, ramp_time=0.1)
    assert ramped.ramp is not None
    # The cos2envelope-based ramp has the same length as the waveform.
    n_samples = ramped.ramp.next(waveform.shape[-1]).shape[-1]
    assert n_samples == waveform.shape[-1]


def test_ramped_epoch_output_ramp_none_path(ao_channel):
    ramped = RampedEpochOutput(name='ramped')
    ao_channel.add_output(ramped)
    waveform = np.ones(100)
    ramped.set_waveform(waveform, ramp_time=None)
    # When ramp_time is None, the ramp array is all zeros (no attenuation
    # change applied later in write_next_samples).
    ramp_samples = ramped.ramp.next(100)
    np.testing.assert_array_equal(ramp_samples, np.zeros(100))
