import pytest

import json

import numpy as np

from psi.data.sinks.api import TextStore
from psidata.api import Recording


@pytest.fixture
def store(tmpdir):
    store = TextStore()
    store.set_base_path(tmpdir)
    return store


def test_continuous(store):
    with pytest.raises(NotImplementedError):
        store.create_ai_continuous('signal', 100e3, 'd', {})


def test_epochs(store, data_generator):
    isi = 5
    metadata = {
        'input_channel': 'ai1',
    }
    epochs = store.create_ai_epochs('epochs', 100e3, 'd', metadata)
    for epochs in data_generator.iter_epochs(isi):
        store.process_ai_epochs('epochs', epochs)

    source = store.get_source('epochs')
    with pytest.raises(TypeError):
        groups = source.get_epoch_groups('stim')

    # Ensure metadata is written out to file
    assert source.dirty
    source.flush()
    assert not source.dirty

    recording = Recording(store.base_path)
    assert not hasattr(recording, 'epochs')
    assert hasattr(recording, 'epochs_metadata')
    sidecar = store.base_path / 'epochs_metadata.json'
    assert json.loads(sidecar.read_text()) == metadata

    assert recording.epochs_metadata.shape == (10, 4)
    np.testing.assert_array_equal(
        recording.epochs_metadata['t0'],
        np.arange(0, data_generator.n_iter * isi, isi)
    )
