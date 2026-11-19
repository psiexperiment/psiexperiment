import subprocess
import sys
import textwrap

import pytest

import numpy as np

from psi.data.sinks.api import ZarrStore
from psidata.api import Recording


@pytest.fixture
def store(tmpdir):
    store = ZarrStore()
    store.set_base_path(tmpdir, is_temp=False)
    yield store
    # Rather than leave anything held for the flush at exit.
    store.flush()


def test_continuous(store, data_generator):
    n_seconds = 10
    metadata = {
        'input_channel': 'ai1',
    }
    store.create_ai_continuous('signal', data_generator.fs, 'd', metadata)

    for samples in data_generator.iter_continuous():
        store.process_ai_continuous('signal', samples)
    generated = np.concatenate(data_generator.generated, axis=-1)
    store.flush()

    recording = Recording(store.base_path)
    assert recording.signal.fs == data_generator.fs
    assert recording.signal.duration == n_seconds
    np.testing.assert_array_equal(generated, recording.signal[:])

    assert recording.signal.array.attrs['input_channel'] == 'ai1'


def test_continuous_multichannel_chunk_shape(store):
    # Regression test: creating a multi-channel continuous array used to
    # pass a bare `None` inside the `chunks` tuple to mean "one chunk
    # spanning the whole channel axis". That's a zarr-version-sensitive
    # convention (unsupported by some zarr array-creation code paths, where
    # it surfaces as a confusing "'NoneType' object is not iterable" deep
    # inside zarr internals), so the channel axis chunk size should be
    # passed explicitly instead.
    metadata = {'input_channel': 'ai1', 'n_channels': 4}
    store.create_ai_continuous('signal', fs=100e3, dtype='d', metadata=metadata)
    zarray = store._stores['signal']
    assert zarray.shape == (4, 0)
    assert zarray.chunks[0] == 4


def test_epochs(store, data_generator):
    isi = 5
    metadata = {
        'input_channel': 'ai1',
    }
    store.create_ai_epochs('epochs', data_generator.fs, 'd', metadata)
    for epochs in data_generator.iter_epochs(isi):
        store.process_ai_epochs('epochs', epochs)

    # Epochs are stored concatenated along the time axis.
    generated = np.concatenate(data_generator.generated, axis=-1)
    np.testing.assert_array_equal(
        np.asarray(store._stores['epochs'].data), generated)

    # Ensure metadata is written out to file on flush.
    assert store._stores['epochs'].dirty
    store.flush()
    assert not store._stores['epochs'].dirty

    recording = Recording(store.base_path)
    assert hasattr(recording, 'epochs')
    assert hasattr(recording, 'epochs_metadata')

    assert recording.epochs.fs == data_generator.fs
    np.testing.assert_array_equal(recording.epochs[:], generated)

    assert recording.epochs_metadata.shape == (10, 4)
    np.testing.assert_array_equal(
        recording.epochs_metadata['t0'],
        np.arange(0, data_generator.n_iter * isi, isi)
    )


def test_continuous_chunk_write_retry(store, data_generator, monkeypatch):
    # Regression test: a transient PermissionError on a chunk write used to
    # retry `zarr.Array.append` as a whole. The resize from the failed attempt
    # was already committed, so the retry grew the array again, shifting all
    # later data by one block.
    from zarr.storage import LocalStore
    set_chunk = LocalStore.set
    failures = {'remaining': 1}

    async def flaky_set(self, key, value):
        if '/c/' in f'/{key}' and failures['remaining']:
            failures['remaining'] -= 1
            raise PermissionError(5, 'Access is denied')
        return await set_chunk(self, key, value)

    monkeypatch.setattr(LocalStore, 'set', flaky_set)
    store.create_ai_continuous('signal', data_generator.fs, 'd', {})
    for samples in data_generator.iter_continuous():
        store.process_ai_continuous('signal', samples)

    assert failures['remaining'] == 0
    generated = np.concatenate(data_generator.generated, axis=-1)
    np.testing.assert_array_equal(generated, store._stores['signal'][:])


def count_writes(monkeypatch):
    '''
    Record the keys of the chunk files and the array metadata (`zarr.json`)
    written from now on.
    '''
    from zarr.storage import LocalStore
    set_key = LocalStore.set
    writes = {'chunks': [], 'metadata': []}

    async def counting_set(self, key, value):
        if '/c/' in f'/{key}':
            writes['chunks'].append(key)
        elif key.endswith('zarr.json'):
            writes['metadata'].append(key)
        return await set_key(self, key, value)

    monkeypatch.setattr(LocalStore, 'set', counting_set)
    return writes


def append_blocks(store, name, n_samples, block=1000, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for i in range(0, n_samples, block):
        x = rng.uniform(size=min(block, n_samples - i))
        store.process_ai_continuous(name, x)
        blocks.append(x)
    return np.concatenate(blocks)


def test_continuous_writes_whole_chunks(store, monkeypatch):
    # Writing each block as it came resized the array and rewrote the chunk
    # it landed in every time, which was too slow for a sound card's inputs
    # to keep up. Only full chunks are written until the store is flushed.
    store.create_ai_continuous('signal', 100e3, 'd', {})
    writes = count_writes(monkeypatch)
    chunk = store._stores['signal'].chunks[-1]
    # 2.5 chunks in 328 blocks.
    generated = append_blocks(store, 'signal', int(chunk * 2.5))
    assert len(writes['chunks']) == 2
    # The array is resized (rewriting its metadata) as each chunk is
    # written, not for every block.
    assert 1 <= len(writes['metadata']) <= 2

    store.flush()
    assert len(writes['chunks']) == 3
    assert len(writes['metadata']) <= 3
    recording = Recording(store.base_path)
    np.testing.assert_array_equal(recording.signal[:], generated)


def test_continuous_read_during_experiment(store):
    # What hasn't filled a chunk yet is written out when the store is read,
    # so readers in the experiment see everything appended, and later
    # appends carry on from there.
    store.create_ai_continuous('signal', 100e3, 'd', {})
    chunk = store._stores['signal'].chunks[-1]
    first = append_blocks(store, 'signal', int(chunk * 1.5), seed=1)
    source = store.get_source('signal')
    assert source.shape == (len(first),)
    np.testing.assert_array_equal(source[:], first)

    second = append_blocks(store, 'signal', chunk, seed=2)
    generated = np.concatenate((first, second))
    np.testing.assert_array_equal(source[:], generated)
    store.flush()
    np.testing.assert_array_equal(Recording(store.base_path).signal[:],
                                  generated)


def test_continuous_multichannel_whole_chunks(store):
    store.create_ai_continuous('signal', 100e3, 'd', {'n_channels': 2})
    chunk = store._stores['signal'].chunks[-1]
    rng = np.random.default_rng(0)
    blocks = [rng.uniform(size=(2, 1000)) for _ in range(chunk * 3 // 1000)]
    for x in blocks:
        store.process_ai_continuous('signal', x)
    store.flush()
    np.testing.assert_array_equal(Recording(store.base_path).signal[:],
                                  np.concatenate(blocks, axis=-1))


def test_continuous_written_through_after_flush(store):
    # Anything arriving after the experiment ends (when the store is
    # flushed) is written as it comes.
    store.create_ai_continuous('signal', 100e3, 'd', {})
    first = append_blocks(store, 'signal', 1500, seed=1)
    store.flush()
    second = append_blocks(store, 'signal', 1500, seed=2)
    np.testing.assert_array_equal(Recording(store.base_path).signal[:],
                                  np.concatenate((first, second)))


def test_continuity_check_counts_unwritten_samples(store, caplog):
    # The continuity check compares each block's s0 with everything
    # appended, including what is still waiting to fill a chunk.
    from psiaudio.pipeline import PipelineData
    store.create_ai_continuous('signal', 100e3, 'd', {})
    for i in range(5):
        x = PipelineData(np.zeros(1000), fs=100e3, s0=i * 1000)
        store.process_ai_continuous('signal', x)
    assert 'NIDAQ_DATA_GAP' not in caplog.text


def test_continuous_flushed_at_exit(tmp_path):
    # If the experiment never reaches experiment_end, what hasn't filled a
    # chunk is written when Python exits. That has to happen before Python
    # shuts down the thread pool zarr writes through, which it does before
    # any atexit callback.
    script = textwrap.dedent(f'''
        import enaml
        import numpy as np
        with enaml.imports():
            from psi.data.sinks.api import ZarrStore
        store = ZarrStore()
        store.set_base_path({str(tmp_path)!r}, is_temp=False)
        store.create_ai_continuous('signal', 100e3, 'd', {{}})
        store.process_ai_continuous('signal', np.arange(5000.0))
    ''')
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr
    assert 'Traceback' not in proc.stderr, proc.stderr
    np.testing.assert_array_equal(Recording(tmp_path).signal[:],
                                  np.arange(5000.0))
