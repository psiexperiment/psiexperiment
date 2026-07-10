import pytest

import numpy as np

from psiaudio.pipeline import PipelineData


class DataGenerator:

    def __init__(self, fs=100e3, n_iter=10, sample_duration=1):
        self.fs = fs
        self.n_iter = n_iter
        self.sample_duration = sample_duration
        self.n_samples = int(round(sample_duration * fs))

    def _iter(self):
        self.generated = []
        for _ in range(self.n_iter):
            samples = np.random.uniform(size=self.n_samples)
            self.generated.append(samples)
            yield samples


    def iter_continuous(self):
        yield from self._iter()

    def iter_epochs(self, isi):
        # Mimics what capture_epoch delivers: PipelineData segments whose
        # metadata merges the epoch info (t0, duration) with the stimulus
        # metadata.
        for i, samples in enumerate(self._iter()):
            metadata = {
                't0': isi * i,
                'duration': self.sample_duration,
                'frequency': 1e3,
                'stim': i,
            }
            epoch = PipelineData(samples, fs=self.fs,
                                 s0=int(round(isi * i * self.fs)),
                                 metadata=metadata)
            yield [epoch]


@pytest.fixture
def data_generator():
    return DataGenerator()
