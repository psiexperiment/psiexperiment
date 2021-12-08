import pytest

import numpy as np


class DataGenerator:

    def __init__(self, fs=100e3, n_iter=10, sample_duration=1):
        self.fs = fs
        self.n_iter = n_iter
        self.sample_duration = sample_duration
        self.n_samples = int(round(sample_duration * fs))

    def _iter(self):
        self.generated = []
        for i in range(self.n_iter):
            samples = np.random.uniform(size=self.n_samples)
            self.generated.append(samples)
            yield samples


    def iter_continuous(self):
        yield from self._iter()

    def iter_epochs(self, isi):
        for i, samples in enumerate(self._iter()):
            signal = {
                'signal': samples,
                'info': {
                    't0': isi * i,
                    'duration': self.sample_duration,
                    'metadata': {'frequency': 1e3, 'stim': i},
                }
            }
            yield [signal]


@pytest.fixture
def data_generator():
    return DataGenerator()
