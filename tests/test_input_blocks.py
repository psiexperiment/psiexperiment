"""Unit tests for psi.controller.input Input subclasses.

These exercise pure-Python behavior of the Declarative wrappers: computed
properties, validation, and (where it can be done without a workbench) the
callback wiring. They do NOT require a workbench, IO manifest, or running
engine -- just a channel as the source via the `ai_channel` fixture.
"""
import numpy as np
import pytest

from psiaudio.calibration import FlatCalibration

from psi.controller.input import (
    Bitmask, Blocked, CalibratedInput, Decimate, DecimateTo, Delay, Downsample,
    EventRate, ExtractEpochs, IIRFilter, RMS, SPL, Threshold,
)


# -------- IIRFilter.Wn --------

def test_iir_filter_wn_lowpass():
    f = IIRFilter(btype='lowpass', f_lowpass=1000, f_highpass=10)
    assert f.Wn == 1000


def test_iir_filter_wn_highpass():
    f = IIRFilter(btype='highpass', f_lowpass=1000, f_highpass=10)
    assert f.Wn == 10


def test_iir_filter_wn_bandpass_tuple():
    f = IIRFilter(btype='bandpass', f_lowpass=1000, f_highpass=10)
    assert f.Wn == (10, 1000)


def test_iir_filter_wn_bandstop_tuple():
    f = IIRFilter(btype='bandstop', f_lowpass=1000, f_highpass=10)
    assert f.Wn == (10, 1000)


# -------- Threshold.function --------

@pytest.mark.parametrize('mode,th,inputs,expected', [
    ('>',  0.0, [-1, 0, 1], [False, False, True]),
    ('>=', 0.0, [-1, 0, 1], [False, True, True]),
    ('<',  0.0, [-1, 0, 1], [True, False, False]),
    ('<=', 0.0, [-1, 0, 1], [True, True, False]),
])
def test_threshold_function(mode, th, inputs, expected):
    t = Threshold(threshold=th, mode=mode)
    fn = t.function
    out = [bool(fn(x)) for x in inputs]
    assert out == expected


# -------- Bitmask.function --------

@pytest.mark.parametrize('bit,values,expected', [
    (0, [0, 1, 2, 3], [False, True, False, True]),
    (1, [0, 1, 2, 3], [False, False, True, True]),
    (2, [0, 4, 5, 8], [False, True, True, False]),
])
def test_bitmask_function(bit, values, expected):
    b = Bitmask(bit=bit)
    fn = b.function
    out = np.array(values, dtype=np.int32)
    np.testing.assert_array_equal(fn(out), expected)


# -------- CalibratedInput --------

def test_calibrated_input_output_is_flat_zero_db(ai_channel):
    ci = CalibratedInput()
    ai_channel.add_input(ci)
    cal = ci.calibration
    # The output calibration is flat with 0 dB sensitivity.
    assert isinstance(cal, FlatCalibration)
    assert cal.sensitivity == 0


# -------- fs / q math --------

def test_downsample_fs(ai_channel):
    d = Downsample(q=10)
    ai_channel.add_input(d)
    assert d.fs == ai_channel.fs / 10


def test_decimate_fs(ai_channel):
    d = Decimate(q=4)
    ai_channel.add_input(d)
    assert d.fs == ai_channel.fs / 4


def test_decimate_to_q_picks_floor(ai_channel):
    # source fs is 100e3; target 30e3 -> q = floor(100/30) = 3 -> fs = 33.3e3
    d = DecimateTo(target_fs=30e3)
    ai_channel.add_input(d)
    assert d.q == 3
    assert d.fs == ai_channel.fs / 3


def test_rms_fs(ai_channel):
    rms = RMS(duration=0.01)
    ai_channel.add_input(rms)
    n = round(0.01 * ai_channel.fs)
    assert rms.fs == ai_channel.fs / n


def test_event_rate_samples(ai_channel):
    er = EventRate(block_size=1.0, block_step=0.25)
    ai_channel.add_input(er)
    assert er.block_size_samples == round(1.0 * ai_channel.fs)
    assert er.block_step_samples == round(0.25 * ai_channel.fs)
    assert er.fs == ai_channel.fs / er.block_step_samples


# -------- Validation --------

def test_blocked_duration_must_be_positive(ai_channel):
    b = Blocked(duration=0)
    ai_channel.add_input(b)
    with pytest.raises(ValueError, match='must be > 0'):
        b.configure_callback()


def test_extract_epochs_infinite_size_raises(ai_channel):
    e = ExtractEpochs(epoch_size=np.inf)
    ai_channel.add_input(e)
    with pytest.raises(ValueError, match='infinite epoch size'):
        e.configure_callback()


# -------- ExtractEpochs.duration --------

def test_extract_epochs_duration_sum(ai_channel):
    e = ExtractEpochs(epoch_size=0.5, prestim_time=0.1, poststim_time=0.2)
    ai_channel.add_input(e)
    assert e.duration == pytest.approx(0.8)


def test_extract_epochs_duration_notifies_on_changes(ai_channel):
    e = ExtractEpochs(epoch_size=0.5)
    ai_channel.add_input(e)
    observed = []
    # ExtractEpochs._observe_* call `self.notify('duration', self.duration)`,
    # which delivers the value directly to the callback (not a change dict).
    e.observe('duration', lambda value: observed.append(value))
    e.epoch_size = 1.0
    e.prestim_time = 0.05
    e.poststim_time = 0.05
    # We should have received at least one notification per change.
    assert len(observed) >= 3
    assert e.duration == pytest.approx(1.1)


# -------- SPL --------

def test_spl_function_uses_sensitivity(ai_channel):
    spl = SPL()
    ai_channel.add_input(spl)
    # Calibration is FlatCalibration.as_attenuation() with sensitivity=0.
    # SPL.function: x -> db(x) + sens. With sens=0, SPL(1.0) == 0.
    fn = spl.function
    assert fn(1.0) == pytest.approx(0)


# -------- Delay --------

def test_delay_callback_validation(ai_channel):
    d = Delay(delay=0.001)
    ai_channel.add_input(d)
    # The configure_callback only validates that source/fs/inputs resolve;
    # there's no negative-delay branch but we can at least round-trip the
    # property without raising.
    assert d.fs == ai_channel.fs


# -------- AutoThreshold._set_auto_th --------

def test_auto_threshold_sets_current_when_unset(ai_channel):
    from psi.controller.input import AutoThreshold
    at = AutoThreshold(n=5, baseline=10)
    ai_channel.add_input(at)
    # current_th starts NaN; setting auto_th propagates to current_th.
    assert np.isnan(at.current_th)
    at._set_auto_th(42.0)
    assert at.auto_th == 42.0
    assert at.current_th == 42.0


def test_auto_threshold_does_not_overwrite_user_override(ai_channel):
    from psi.controller.input import AutoThreshold
    at = AutoThreshold(n=5, baseline=10)
    ai_channel.add_input(at)
    at.current_th = 7.5  # user-set override
    at._set_auto_th(42.0)
    # auto_th updates, but the user's current_th is preserved.
    assert at.auto_th == 42.0
    assert at.current_th == 7.5


# -------- RejectEpochs.status_cb --------

def test_reject_epochs_status_math(ai_channel, monkeypatch):
    from psi.controller.input import RejectEpochs
    # Make deferred_call synchronous so we can assert on counters in-line.
    monkeypatch.setattr(
        'psi.controller.input.deferred_call',
        lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )
    re = RejectEpochs(threshold=1.0, mode='absolute value', running_N=4)
    ai_channel.add_input(re)
    # First batch: 3 epochs, 1 rejected (66.7% accepted).
    re.status_cb([True, False, True])
    assert re.total == 3
    assert re.rejects == 1
    assert re.reject_percent == pytest.approx(100 * 1 / 3)
    # Second batch: 2 more epochs, both rejected.
    re.status_cb([False, False])
    assert re.total == 5
    assert re.rejects == 3
    assert re.reject_percent == pytest.approx(100 * 3 / 5)
    # running_N=4, last 4 accepted flags: [False, True, False, False] -> 1/4 accepted.
    # running_reject_percent = (1 - 1/4) * 100 = 75.
    assert re.running_reject_percent == pytest.approx(75.0)


# -------- Detrend default --------

def test_detrend_default_mode(ai_channel):
    from psi.controller.input import Detrend
    d = Detrend()
    ai_channel.add_input(d)
    # Default mode is the first Enum entry: 'constant'.
    assert d.mode == 'constant'


# -------- Edges defaults --------

def test_edges_defaults(ai_channel):
    from psi.controller.input import Edges
    e = Edges()
    ai_channel.add_input(e)
    assert e.initial_state == 0
    assert e.debounce == 2
    assert e.detect == 'rising'
    assert e.min_events == 0


# -------- Accumulate defaults --------

def test_accumulate_defaults(ai_channel):
    from psi.controller.input import Accumulate
    a = Accumulate(n=10)
    ai_channel.add_input(a)
    assert a.n == 10
    assert a.axis == -1
    assert a.newaxis is False


# -------- Capture defaults --------

def test_capture_defaults(ai_channel):
    from psi.controller.input import Capture
    c = Capture()
    ai_channel.add_input(c)
    # queue starts as an empty deque.
    assert len(c.queue) == 0
    assert c.start_event == ''


# -------- EventsToInfo defaults --------

def test_events_to_info_defaults(ai_channel):
    from psi.controller.input import EventsToInfo
    # ai_channel is a continuous source, but for this Declarative test we
    # only care about default Atom values, not callback wiring.
    eti = EventsToInfo()
    assert eti.trigger_edge == 'rising'
    assert eti.base_info == {}


# -------- Discard fs is source fs --------

def test_discard_fs_unchanged(ai_channel):
    from psi.controller.input import Discard
    d = Discard(duration=0.1)
    ai_channel.add_input(d)
    # Discard preserves sample rate; only truncates a chunk from the front.
    assert d.fs == ai_channel.fs


# -------- Derivative initial_value --------

def test_derivative_initial_value(ai_channel):
    from psi.controller.input import Derivative
    d = Derivative(initial_value=2.5)
    ai_channel.add_input(d)
    assert d.initial_value == 2.5
    assert d.fs == ai_channel.fs
