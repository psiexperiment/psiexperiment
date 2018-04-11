import pytest
import numpy as np

from psi.data.plots import SignalBuffer


@pytest.fixture
def sb():
    return SignalBuffer(fs=100, size=10)


def test_buffer_bounds(sb):
    assert sb.get_time_ub() == 0
    assert sb.get_time_lb() == 0

    with pytest.raises(IndexError):
        result = sb.get_range(0, 5)

    data1 = np.random.uniform(size=100)
    sb.append_data(data1)
    assert sb.get_time_ub() == 1
    assert sb.get_time_lb() == 0

    result = sb.get_range(0, 1)
    assert np.all(result == data1)

    data2 = np.random.uniform(size=900)
    sb.append_data(data2)
    assert sb.get_time_ub() == 10
    assert sb.get_time_lb() == 0

    result = sb.get_range(0, 1)
    assert np.all(result == data1)
    result = sb.get_range(0, 10)
    assert np.all(result == np.concatenate((data1, data2)))

    data3 = np.random.uniform(size=100)
    sb.append_data(data3)
    assert sb.get_time_ub() == 11
    assert sb.get_time_lb() == 1

    result = sb.get_range(1, 11)
    assert np.all(result == np.concatenate((data2, data3)))

    result = sb.get_range(10, 11)
    assert np.all(result == data3)

    result = sb.get_latest(-1)
    assert np.all(result == data3)


def test_buffer_overfill(sb):
    data = np.random.uniform(size=10000)
    sb.append_data(data)
    assert sb.get_time_lb() == 90
    assert sb.get_time_ub() == 100
    with pytest.raises(IndexError):
        sb.get_range(85, 90)
    result = sb.get_range(95)
    assert np.all(result == data[-500:])


def test_buffer_filled(sb):
    result = sb.get_range_filled(0, 1, np.nan)
    assert result.shape == (100,)
    assert np.all(np.isnan(result))

    data1 = np.random.uniform(size=500)
    sb.append_data(data1)
    result = sb.get_range_filled(2.5, 7.5, np.nan)
    assert result.shape == (500,)
    assert np.all(np.isnan(result[250:]))
    assert np.all(result[:250] == data1[250:])

    data2 = np.random.uniform(size=1000)
    sb.append_data(data2)
    result = sb.get_range_filled(2.5, 7.5, np.nan)
    assert np.all(np.isnan(result[:250]))
    assert np.all(result[250:] == data2[:250])


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
