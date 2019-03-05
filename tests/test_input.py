import numpy as np
import pytest

from psi.controller.input import (accumulate, blocked, concatenate, coroutine,
                                  InputData)


@pytest.fixture
def data():
    return []


@pytest.fixture
def pipeline(data):
    def append_data(d):
        nonlocal data
        if d is not Ellipsis:
            data.append(d)

    return blocked(10,
                    accumulate(4, 0, True, None,
                               append_data).send)


def test_input_data():
    expected_array = np.concatenate((
        np.zeros(shape=5),
        np.ones(shape=5),
        np.random.uniform(size=5),
    ))
    expected_metadata = {'t0': 0}
    expected_input = InputData(expected_array, expected_metadata)

    input_data = [
        InputData(expected_array[0:5], metadata=expected_metadata),
        InputData(expected_array[5:10], metadata=expected_metadata),
        InputData(expected_array[10:15], metadata=expected_metadata),
    ]

    result = concatenate(input_data)
    assert expected_input.metadata == result.metadata
    assert np.array_equal(expected_input, result)

    input_mean = np.mean(input_data, axis=0)
    print(input_mean.metadata)
    assert False


def test_pipeline(data, pipeline):
    expected = []
    for i in range(8):
        d = InputData(np.random.uniform(size=5), {'n': 5})
        pipeline.send(d)
        expected.append(d[np.newaxis])
    expected = concatenate(expected, axis=-1).reshape((-1, 10))

    assert len(data) == 1
    assert data[0].shape == (4, 10)
    assert np.array_equal(expected, data[0])
    assert data[0].metadata == expected.metadata


def test_pipeline_flush(data, pipeline):
    expected = []
    for i in range(16):
        d = InputData(np.random.uniform(size=5), {'n': 5})
        pipeline.send(d)
        expected.append(d[np.newaxis])
        if i == 6:
            pipeline.send(Ellipsis)
            expected = []

    expected = concatenate(expected[:8], axis=-1).reshape((-1, 10))
    assert len(data) == 1
    assert data[0].shape == (4, 10)
    assert np.array_equal(expected, data[0])
    assert data[0].metadata == expected.metadata
