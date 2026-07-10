import pytest

import json

import numpy as np

from psi.data.sinks.api import TextStore
from psidata.api import Recording


@pytest.fixture
def store(tmpdir):
    store = TextStore()
    store.set_base_path(tmpdir, is_temp=False)
    return store


def test_continuous(store):
    with pytest.raises(NotImplementedError):
        store.create_ai_continuous('signal', 100e3, 'd', {})


def test_epochs(store, data_generator):
    isi = 5
    metadata = {
        'input_channel': 'ai1',
    }
    store.create_ai_epochs('epochs', 100e3, 'd', metadata)
    for epochs in data_generator.iter_epochs(isi):
        store.process_ai_epochs('epochs', epochs)

    # TextStore only saves epoch metadata, not the signal itself.
    with pytest.raises(TypeError):
        store._stores['epochs'].get_epoch_groups('stim')

    # get_source returns the accumulated metadata as a DataFrame.
    source = store.get_source('epochs')
    assert source.shape == (data_generator.n_iter, 4)

    # Ensure metadata is written out to file on flush.
    assert store._stores['epochs'].dirty
    store.flush()
    assert not store._stores['epochs'].dirty

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
