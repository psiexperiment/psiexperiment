import logging
log = logging.getLogger(__name__)

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
    df = table.todataframe()
    if decode:
        for c in table.cols:
            if table[c].dtype.char == 'S':
                df[c] = df[c].str.decode('utf8')

    if archive:
        df.to_csv(csv_path, index=False)
    return df


def get_unique_columns(df, exclude=None):
    return [c for c in df if (len(df[c].unique()) > 1) and (c not in exclude)]


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

    def __getitem__(self, slice):
        return self.array[slice]

    @property
    def shape(self):
        return self.array.shape


def get_epoch_groups(epoch, epoch_md, groups):
    # Used by speaker_calibration
    fs = epoch.attrs['fs']
    df = epoch_md.todataframe()
    df['samples'] = np.round(df['duration']*fs).astype('i')
    df['offset'] = df['samples'].cumsum() - df.loc[0, 'samples']

    epochs = {}
    for keys, g_df in df.groupby(groups):
        data = []
        for _, row in g_df.iterrows():
            o = row['offset']
            s = row['samples']
            d = epoch[o:o+s][np.newaxis]
            data.append(d)
        epochs[keys] = np.concatenate(data, axis=0)

    return epochs
