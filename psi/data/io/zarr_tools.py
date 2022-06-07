import logging
log = logging.getLogger(__name__)

import functools

import zarr

from . import Signal


# Max size of LRU cache
MAXSIZE = 1024


class ZarrSignal(Signal):

    @classmethod
    def from_path(cls, path):
        path = path.with_suffix('.zarr')
        array = zarr.open(store=str(path), mode='r')
        return cls(array)

    def __init__(self, array):
        self.array = array

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
