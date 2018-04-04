import pytest
import numpy as np

from psi.data.plots import SignalBuffer


@pytest.fixture
def sb():
    return SignalBuffer(fs=100, size=10)


def test_buffer_bounds(sb):
    assert sb.get_time_ub() == 0
    assert sb.get_time_lb() == 0

    result = sb.get_range(0, 5)
    assert np.all(np.isnan(result))

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
