import pytest
import numpy as np

from psi.data.plots import SignalBuffer


@pytest.fixture(params=[None, 4])
def n_channels(request):
    return request.param


@pytest.fixture(params=[100, 195312.5])
def fs(request):
    return request.param


@pytest.fixture(params=[1, 10])
def duration(request):
    return request.param


@pytest.fixture
def sb(fs, duration, n_channels):
    return SignalBuffer(fs=fs, size=duration, n_channels=n_channels)


def test_buffer_get_latest(fs, duration, n_channels, sb):
    if n_channels is None:
        write_samples = np.random.uniform(size=201858)
    else:
        write_samples = np.random.uniform(size=(n_channels, 201858))

    sb.append_data(write_samples)
    read_samples = sb.get_latest(-1, 0)

    n = int(np.ceil(fs))
    assert read_samples.shape[-1] == n
    np.testing.assert_array_equal(read_samples, write_samples[..., -n:])


def test_buffer_bounds(fs, duration, n_channels, sb):
    assert sb.get_time_ub() == 0
    assert sb.get_time_lb() == 0

    with pytest.raises(IndexError):
        result = sb.get_range(0, 5)

    sizes = (100, 900, 100)
    cumulative_size = 0

    if n_channels is not None:
        cumulative_data = np.zeros((n_channels, 0))
    else:
        cumulative_data = np.zeros(0)

    cumulative_chunks = []
    cumulative_bounds = []

    max_n = int(np.ceil(duration * fs))

    for size in sizes:
        if n_channels is not None:
            shape = (n_channels, size)
        else:
            shape = size

        data = np.random.uniform(size=shape)
        sb.append_data(data)

        segment_lb = cumulative_size / fs
        cumulative_size += size
        segment_ub = cumulative_size / fs

        cumulative_chunks.append(data)
        cumulative_bounds.append((segment_lb, segment_ub))

        cumulative_data = np.concatenate((cumulative_data, data), axis=-1)

        ub = segment_ub
        lb = max(0, ub - duration)
        assert sb.get_time_ub() == ub
        assert sb.get_time_lb() == lb

        if segment_lb >= sb.get_time_lb():
            result = sb.get_latest(segment_lb-segment_ub)
            np.testing.assert_array_equal(result, data[..., -max_n:])

        result = sb.get_latest(lb-ub)
        np.testing.assert_array_equal(result, cumulative_data[..., -max_n:])

        if lb > 0:
            with pytest.raises(IndexError):
                sb.get_range(0, ub)
        else:
            result = sb.get_range(0, ub)
            np.testing.assert_array_equal(result, cumulative_data)

            if ub < 1:
                with pytest.raises(IndexError):
                    sb.get_range(0, 1)
            else:
                result = sb.get_range(0, 1)
                n = int(np.ceil(fs))
                np.testing.assert_array_equal(result, cumulative_data[..., :n])

        for (lb, ub), chunk in zip(cumulative_bounds, cumulative_chunks):
            if lb < sb.get_time_lb():
                continue
            result = sb.get_range(lb, ub)
            np.testing.assert_array_equal(result, chunk[..., -max_n:])

        for offset in (0, 2, 4):
            if ub < offset:
                continue
            n = int(np.ceil(ub - max(offset, sb.get_time_lb())) * fs)
            result = sb.get_range_filled(offset, ub, np.nan)
            expected = np.full_like(result, np.nan)
            expected[..., -n:] = cumulative_data[..., -n:]
            np.testing.assert_array_equal(result, expected)


def test_buffer_invalidate_zero(sb):
    sb.invalidate(0)


@pytest.mark.skip
def test_buffer_invalidate(sb):
    data1 = np.random.uniform(size=500)
    sb.append_data(data1)
    assert sb.get_time_lb() == 0
    assert sb.get_time_ub() == 5
    result = sb.get_range(0, 2.5)
    assert np.all(result == data1[:250])
    result = sb.get_range(2.5, 5)
    assert np.all(result == data1[250:])

    sb.invalidate(2.5)
    assert sb.get_time_ub() == 2.5
    with pytest.raises(IndexError):
        result = sb.get_range(2.5, 5)
    result = sb.get_range(0, 2.5)
    assert np.all(result == data1[:250])

    data2 = np.random.uniform(size=100)
    sb.append_data(data2)
    assert sb.get_time_ub() == 3.5
    result = sb.get_range(2.5, 3.5)
    assert np.all(result == data2)
    result = sb.get_range(0, 3.5)
    expected = np.concatenate((data1[:250], data2), axis=-1)
    assert np.all(result == expected)

    assert sb.get_samples_lb() == 0
    assert sb.get_samples_ub() == 350

    data3 = np.random.uniform(size=900)
    sb.append_data(data3)
    assert sb.get_time_ub() == 12.5
    assert sb.get_time_lb() == 2.5
    result = sb.get_range(2.5, 12.5)
    expected = np.concatenate((data2, data3), axis=-1)
    assert np.all(result == expected)

    sb.invalidate(10)
    assert sb.get_time_ub() == 10
    assert sb.get_time_lb() == 2.5

    result = sb.get_range(2.5, 10)
    expected = np.concatenate((data2, data3[:650]), axis=-1)

    with pytest.raises(IndexError):
        sb.get_range(1.5, 10)

    sb.invalidate_samples(250)
    assert sb.get_time_ub() == 2.5
    assert sb.get_time_lb() == 2.5

    data4 = np.random.uniform(size=100)
    sb.append_data(data4)
    assert sb.get_time_ub() == 3.5
    assert sb.get_time_lb() == 2.5
    result = sb.get_range(2.5, 3.5)
    assert np.all(result == data4)

    with pytest.raises(IndexError):
        sb.get_range(1.5, 2.5)

    with pytest.raises(IndexError):
        sb.get_range(3.5, 4.5)

    result = sb.get_range()
    assert np.all(result == data4)


@pytest.mark.skip
def test_buffer_invalidate_past_end(sb):
    assert sb.get_samples_lb() == 0
    assert sb.get_samples_ub() == 0
    sb.invalidate_samples(5000)
    assert sb.get_samples_lb() == 0
    assert sb.get_samples_ub() == 0

    data = np.random.uniform(size=100)
    sb.append_data(data)
    assert sb.get_samples_lb() == 0
    assert sb.get_samples_ub() == 100
    sb.invalidate_samples(5000)
    assert sb.get_samples_lb() == 0
    assert sb.get_samples_ub() == 100

    data = np.random.uniform(size=4900)
    sb.append_data(data)
    assert sb.get_samples_lb() == 4000
    assert sb.get_samples_ub() == 5000

    sb.invalidate_samples(5000)
    assert sb.get_samples_lb() == 4000
    assert sb.get_samples_ub() == 5000

    sb.invalidate_samples(4999)
    assert sb.get_samples_lb() == 4000
    assert sb.get_samples_ub() == 4999


def test_buffer_resize(fs, n_channels):
    duration = 1
    sb = SignalBuffer(fs=fs, size=duration, n_channels=n_channels)
    n_samples = int(duration * fs)
    if n_channels is not None:
        write_samples = np.random.uniform(size=(n_channels, n_samples))
    else:
        write_samples = np.random.uniform(size=n_samples)

    sb.append_data(write_samples)
    samples = sb.get_latest(-duration)
    np.testing.assert_array_equal(samples, write_samples)

    sb.resize(5)
    samples = sb.get_latest(-duration)
    np.testing.assert_array_equal(samples, write_samples)

    sb.resize(20)
    samples = sb.get_latest(-duration)
    np.testing.assert_array_equal(samples, write_samples)

    sb.resize(1)
    samples = sb.get_latest(-1)
    n = int(round(sb._buffer_fs))
    np.testing.assert_array_equal(samples, write_samples[..., -n:])

    sb.resize(2)
    samples = sb.get_latest(-2, fill_value=np.nan)
    assert samples.shape[-1] == int(round(sb._buffer_fs * 2))
    np.testing.assert_array_equal(samples[..., -n:], write_samples[..., -n:])
    assert np.all(np.isnan(samples[..., :n]))


def test_buffer_overfill(fs, n_channels):
    duration = 10
    sb = SignalBuffer(fs=fs, size=duration, n_channels=n_channels)

    n = int(round(fs * 100))
    size = (n_channels, n) if n_channels is not None else (n,)
    data = np.random.uniform(size=size)
    sb.append_data(data)
    assert sb.get_time_lb() == 90
    assert sb.get_time_ub() == 100
    with pytest.raises(IndexError):
        sb.get_range(85, 90)
    result = sb.get_range(95)

    n = int(round(fs * 5))
    np.testing.assert_array_equal(result, data[..., -n:])


def test_buffer_filled(fs, n_channels):
    duration = 5
    sb = SignalBuffer(fs=fs, size=duration, n_channels=n_channels)

    result = sb.get_range_filled(0, 1, np.nan)
    n_samples = int(round(fs))
    if n_channels is not None:
        expected_shape = (n_channels, n_samples)
    else:
        expected_shape = (n_samples,)
    assert result.shape == expected_shape
    assert np.all(np.isnan(result))

    # Append 5 seconds worth of data. Since signal buffer duration is 5, we
    # will be writing the spam from 0 to 5 seconds.
    n_samples = int(np.round(fs * 5))
    if n_channels is not None:
        size = (n_channels, n_samples)
    else:
        size = (n_samples,)
    data1 = np.random.uniform(size=size)
    sb.append_data(data1)

    result = sb.get_range_filled(2.5, 7.5, np.nan)
    n_half = int(np.round(fs * 2.5))
    assert np.all(np.isnan(result[..., n_half:]))
    np.testing.assert_array_equal(result[..., :n_half], data1[..., n_half:])

    # Append 5 more seconds worth of data. Since signal buffer duration is 5,
    # we will be writing the spam from 5 to 10 seconds. The span from 2.5 to 5
    # seconds will be lost.
    data2 = np.random.uniform(size=size)
    sb.append_data(data2)
    result = sb.get_range_filled(2.5, 7.5, np.nan)
    assert np.all(np.isnan(result[..., :n_half-1]))
    # This corrects for some funny sampling rate differences with the TDT
    # hardware.
    n_second_half = result.shape[-1] - n_half
    np.testing.assert_array_equal(result[..., n_half:], data2[..., :n_second_half])


def test_buffer_append():
    sb = SignalBuffer(fs=100, size=1, n_channels=None)
    with pytest.raises(ValueError, match='Appended data must be one-dimensional'):
        samples = np.random.uniform(size=(1, 100))
        sb.append_data(samples)

    sb = SignalBuffer(fs=100, size=1, n_channels=1)
    with pytest.raises(ValueError, match='Appended data must be two-dimensional'):
        samples = np.random.uniform(size=100)
        sb.append_data(samples)

    sb = SignalBuffer(fs=100, size=1, n_channels=1)
    with pytest.raises(ValueError, match='Appended data must have 1 channels.'):
        samples = np.random.uniform(size=(2, 100))
        sb.append_data(samples)
