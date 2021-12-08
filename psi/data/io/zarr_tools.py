import logging
log = logging.getLogger(__name__)

import functools

import zarr

from . import Signal


# Max size of LRU cache
MAXSIZE = 1024


class ZarrSignal(Signal):

    def __init__(self, base_path):
        self.base_path = base_path.with_suffix('.zarr')

    @property
    @functools.lru_cache()
    def array(self):
        return zarr.open(store=str(self.base_path), mode='r')

    @property
    def fs(self):
        return self.array.attrs['fs']

    @property
    def duration(self):
        return self.array.shape[-1]/self.fs

    def __getitem__(self, slice):
        return self.array[slice]

    @property
    def shape(self):
        return self.array.shape
