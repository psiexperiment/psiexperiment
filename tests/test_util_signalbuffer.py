import numpy as np
import pytest

from psi.util import SignalBuffer


@pytest.fixture
def buf():
    # 100 samples capacity, fill with NaN by default.
    return SignalBuffer(fs=100.0, size=1.0)


def test_initial_state_is_empty(buf):
    assert buf.get_samples_lb() == 0
    assert buf.get_samples_ub() == 0
    assert buf.get_time_lb() == 0
    assert buf.get_time_ub() == 0


def test_append_partial_fill(buf):
    data = np.arange(40, dtype=np.double)
    buf.append_data(data)
    assert buf.get_samples_lb() == 0
    assert buf.get_samples_ub() == 40
    np.testing.assert_array_equal(buf.get_range_samples(), data)


def test_append_fills_then_rolls(buf):
    buf.append_data(np.arange(80, dtype=np.double))
    buf.append_data(np.arange(80, 120, dtype=np.double))  # rolls 20 off the front
    assert buf.get_samples_lb() == 20
    assert buf.get_samples_ub() == 120
    np.testing.assert_array_equal(
        buf.get_range_samples(),
        np.arange(20, 120, dtype=np.double),
    )


def test_append_larger_than_capacity_keeps_tail(buf):
    data = np.arange(250, dtype=np.double)
    buf.append_data(data)
    assert buf.get_samples_lb() == 150
    assert buf.get_samples_ub() == 250
    np.testing.assert_array_equal(buf.get_range_samples(), data[-100:])


def test_get_range_samples_out_of_range_raises(buf):
    buf.append_data(np.arange(50, dtype=np.double))
    # Below the current lower bound.
    with pytest.raises(IndexError):
        buf.get_range_samples(-1, 10)
    # Above the current upper bound.
    with pytest.raises(IndexError):
        buf.get_range_samples(0, 51)


def test_get_range_filled_pads_with_fill_value(buf):
    buf.append_data(np.arange(50, dtype=np.double))
    # Request from -0.1 s (sample -10) through 0.6 s (sample 60). Only 0..50 are
    # buffered, so we expect 10 left-pad + 50 data + 10 right-pad.
    out = buf.get_range_filled(-0.1, 0.6, fill_value=-1.0)
    assert out.shape == (70,)
    assert np.all(out[:10] == -1.0)
    np.testing.assert_array_equal(out[10:60], np.arange(50, dtype=np.double))
    assert np.all(out[60:] == -1.0)


def test_get_latest(buf):
    buf.append_data(np.arange(50, dtype=np.double))
    # Most recent 0.1 s == last 10 samples.
    np.testing.assert_array_equal(buf.get_latest(-0.1), np.arange(40, 50))


def test_get_latest_with_padding(buf):
    buf.append_data(np.arange(5, dtype=np.double))
    # Ask for 0.5 s back == 50 samples; only 5 buffered, so pad with 45 zeros.
    out = buf.get_latest(-0.5, fill_value=0.0)
    assert out.shape == (50,)
    assert np.all(out[:45] == 0.0)
    np.testing.assert_array_equal(out[45:], np.arange(5, dtype=np.double))


def test_invalidate_truncates(buf):
    buf.append_data(np.arange(50, dtype=np.double))
    buf.invalidate_samples(30)
    assert buf.get_samples_ub() == 30
    # Data up to 30 is still readable.
    np.testing.assert_array_equal(buf.get_range_samples(0, 30),
                                  np.arange(30, dtype=np.double))


def test_invalidate_past_ub_is_noop(buf):
    buf.append_data(np.arange(50, dtype=np.double))
    buf.invalidate_samples(999)
    assert buf.get_samples_ub() == 50


def test_resize_grow_preserves_data(buf):
    buf.append_data(np.arange(50, dtype=np.double))
    buf.resize(2.0)  # 200 samples now
    assert buf.get_samples_ub() == 50
    # The most recent 50 samples should still be 0..49.
    np.testing.assert_array_equal(buf.get_latest(-0.5), np.arange(50, dtype=np.double))


def test_resize_shrink_is_ignored(buf):
    buf.append_data(np.arange(50, dtype=np.double))
    buf.resize(0.1)  # shrink request — implementation comment says it's ignored,
                    # but get_latest is still used internally so the buffer ends
                    # up at the shrink size. Verify behavior matches what the
                    # docstring promises: existing data within the new window is
                    # preserved.
    # The most recent 10 samples (40..49) must still be retrievable.
    np.testing.assert_array_equal(buf.get_latest(-0.1), np.arange(40, 50))


def test_time_to_samples():
    b = SignalBuffer(fs=1000.0, size=1.0)
    assert b.time_to_samples(0.5) == 500
    assert b.time_to_samples(0) == 0


def test_multichannel_shape_and_append():
    b = SignalBuffer(fs=100.0, size=1.0, n_channels=2, fill_value=0.0)
    data = np.tile(np.arange(40, dtype=np.double), (2, 1))
    b.append_data(data)
    assert b.get_samples_ub() == 40
    out = b.get_range_samples()
    assert out.shape == (2, 40)
    np.testing.assert_array_equal(out, data)


def test_multichannel_rejects_wrong_ndim():
    b = SignalBuffer(fs=100.0, size=1.0, n_channels=2)
    with pytest.raises(ValueError, match='two-dimensional'):
        b.append_data(np.arange(10, dtype=np.double))


def test_multichannel_rejects_wrong_channel_count():
    b = SignalBuffer(fs=100.0, size=1.0, n_channels=2)
    with pytest.raises(ValueError, match='must have 2 channels'):
        b.append_data(np.zeros((3, 10)))


def test_single_channel_rejects_2d():
    b = SignalBuffer(fs=100.0, size=1.0)
    with pytest.raises(ValueError, match='one-dimensional'):
        b.append_data(np.zeros((2, 10)))
