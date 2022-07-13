import logging
log = logging.getLogger(__name__)

from collections import OrderedDict
import functools
from glob import glob
import json
import shutil
from pathlib import Path
import os.path

import bcolz
import numpy as np
import pandas as pd
from scipy import signal

from . import Signal


# Max size of LRU cache
MAXSIZE = 1024


def repair_carray_size(path):
    chunk_wildcard = os.path.join(path, 'data', '*.blp')
    sizes_filename = os.path.join(path, 'meta', 'sizes')
    storage_filename = os.path.join(path, 'meta', 'storage')

    with open(sizes_filename, 'r') as fh:
        sizes = json.load(fh)

    with open(storage_filename, 'r') as fh:
        storage = json.load(fh)

    if sizes['shape'] != [0]:
        raise ValueError('Data seems fine')

    chunklen = storage['chunklen']
    n_chunks = len(glob(chunk_wildcard))
    sizes['shape'] = [n_chunks * chunklen]

    # Backup the file before overwriting.
    shutil.copy(sizes_filename, sizes_filename + '.old')
    with open(sizes_filename, 'w') as fh:
        json.dump(sizes, fh)


def load_ctable_as_df(path, decode=True, archive=True):
    csv_path = f'{path}.csv'
    if os.path.exists(csv_path):
        return pd.io.parsers.read_csv(csv_path)
    table = bcolz.ctable(rootdir=path)
    df = carray_to_dataframe(table)
    if decode:
        for c in table.cols:
            if table[c].dtype.char == 'S':
                df[c] = df[c].str.decode('utf8')

    if archive:
        df.to_csv(csv_path, index=False)
    return df


def carray_to_dataframe(ctable, columns=None, orient='columns'):
    # Right now only legacy bcolz is here.
    if orient == 'index':
        keys = ctable.names
    else:
        keys = ctable.names if columns is None else columns
        columns = None

    # Use a generator here to minimize the number of column copies
    # existing simultaneously in-memory
    df = pd.DataFrame.from_dict(
        OrderedDict((key, ctable[key][:]) for key in keys),
        columns=columns, orient=orient)
    return df


class BcolzSignal(Signal):

    def __init__(self, base_path):
        self.base_path = base_path

    @property
    @functools.lru_cache()
    def array(self):
        return bcolz.carray(rootdir=self.base_path)

    @property
    def fs(self):
        return self.array.attrs['fs']

    @property
    def duration(self):
        return self.array.shape[-1]/self.fs

    @property
    def ndim(self):
        return self.array.ndim

    def __getitem__(self, slice):
        return self.array[slice]

    @property
    def shape(self):
        return self.array.shape
