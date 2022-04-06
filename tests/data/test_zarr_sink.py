import pytest

import numpy as np

from psi.data.sinks.api import ZarrStore
from psi.data.io import Recording


@pytest.fixture
def store(tmpdir):
    store = ZarrStore()
    store.set_base_path(tmpdir)
    return store


def test_continuous(store, data_generator):
    n_seconds = 10
    metadata = {
        'input_channel': 'ai1',
    }
    store.create_ai_continuous('signal', data_generator.fs, 'd', metadata)

    for samples in data_generator.iter_continuous():
        store.process_ai_continuous('signal', samples)
    generated = np.concatenate(data_generator.generated, axis=-1)

    recording = Recording(store.base_path)
    assert recording.signal.fs == data_generator.fs
    assert recording.signal.duration == n_seconds
    np.testing.assert_array_equal(generated, recording.signal[:])

    assert recording.signal.array.attrs['input_channel'] == 'ai1'


def test_epochs(store, data_generator):
    isi = 5
    metadata = {
        'input_channel': 'ai1',
    }
    store.create_ai_epochs('epochs', data_generator.fs, 'd', metadata)
    for epochs in data_generator.iter_epochs(isi):
        store.process_ai_epochs('epochs', epochs)

    # Test grouping by epoch number (stim)
    source = store.get_source('epochs')
    groups = source.get_epoch_groups('stim')
    assert len(groups) == data_generator.n_iter
    for i in range(data_generator.n_iter):
        np.testing.assert_array_equal(
            groups[i],
            data_generator.generated[i][np.newaxis]
        )

    # Test grouping by frequency
    groups = source.get_epoch_groups('frequency')
    assert len(groups) == 1
    np.testing.assert_array_equal(
        groups[1e3].mean(axis=0),
        np.mean(data_generator.generated, axis=0)
    )

    generated = np.concatenate(data_generator.generated, axis=-1)

    # Ensure metadata is written out to file
    assert source.dirty
    source.flush()
    assert not source.dirty

    recording = Recording(store.base_path)
    assert hasattr(recording, 'epochs')
    assert hasattr(recording, 'epochs_metadata')

    assert recording.epochs_metadata.shape == (10, 4)
    np.testing.assert_array_equal(
        recording.epochs_metadata['t0'],
        np.arange(0, data_generator.n_iter * isi, isi)
    )
