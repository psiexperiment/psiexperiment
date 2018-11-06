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


class Signal:

    def get_epochs(self, md, offset, duration, detrend=None, columns='auto'):
        fn = self.get_segments
        return self._get_epochs(fn, md, offset, duration, columns=columns)

    def get_epochs_filtered(self, md, offset, duration, filter_lb, filter_ub,
                            filter_order=1, detrend='constant',
                            pad_duration=10e-3, columns='auto'):
        fn = self.get_segments_filtered
        return self._get_epochs(fn, md, offset, duration, filter_lb, filter_ub,
                                filter_order, detrend, pad_duration,
                                columns=columns)

    def _get_epochs(self, fn, md, *args, columns='auto', **kwargs):
        if columns == 'auto':
            columns = get_unique_columns(md, exclude=['t0'])
        t0 = md['t0'].values
        arrays = [md[c] for c in columns]
        arrays.append(t0)
        df = fn(t0, *args, **kwargs)
        df.index = pd.MultiIndex.from_arrays(arrays, names=columns + ['t0'])
        return df

    def get_segments(self, times, offset, duration, detrend=None):
        times = np.asarray(times)
        indices = np.round((times + offset) * self.fs).astype('i')
        samples = round(duration * self.fs)

        m = (indices >= 0) & ((indices + samples) < self.shape[-1])
        if not m.all():
            i = np.flatnonzero(~m)
            log.warn('Missing epochs %d', i)
            print(f'Missing epochs {i}')

        values = np.concatenate([self[i:i+samples][np.newaxis] \
                                 for i in indices[m]])
        if detrend is not None:
            values = signal.detrend(values, axis=-1, type=detrend)

        t = np.arange(samples)/self.fs + offset
        columns = pd.Index(t, name='time')
        index = pd.Index(times[m], name='t0')
        df = pd.DataFrame(values, index=index, columns=columns)
        return df.reindex(times)

    def _get_segments_filtered(self, fn, offset, duration, filter_lb, filter_ub,
                               filter_order=1, detrend='constant',
                               pad_duration=10e-3):

        Wn = (filter_lb/self.fs, filter_ub/self.fs)
        b, a = signal.iirfilter(filter_order, Wn, btype='band', ftype='butter')
        df = fn(offset-pad_duration, duration+pad_duration, detrend)
        df[:] = signal.filtfilt(b, a, df.values, axis=-1)
        return df.loc[:, offset:offset+duration]

    def get_random_segments(self, n, offset, duration, detrend):
        t_min = -offset
        t_max = self.duration-duration-offset
        times = np.random.uniform(t_min, t_max, size=n)
        return self.get_segments(times, offset, duration, detrend)

    def get_segments_filtered(self, times, *args, **kwargs):
        fn = functools.partial(self.get_segments, times)
        return self._get_segments_filtered(fn, *args, **kwargs)

    def get_random_segments_filtered(self, n, *args, **kwargs):
        fn = functools.partial(self.get_random_segments, n)
        return self._get_segments_filtered(fn, *args, **kwargs)


class BcolzSignal(Signal):

    def __init__(self, base_path):
        self.base_path = base_path

    @property
    @functools.lru_cache(maxsize=MAXSIZE)
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


class Recording:
    pass


class BcolzRecording(Recording):

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        carray_names = set(d.parts[-2] for d in self.base_path.glob('*/meta'))
        ctable_names = set(d.parts[-3] for d in self.base_path.glob('*/*/meta'))
        self.carray_names = carray_names
        self.ctable_names = ctable_names

    def __getattr__(self, attr):
        return self._load_bcolz(attr)

    def __repr__(self):
        carray = ', '.join(self.carray_names)
        ctable = ', '.join(self.ctable_names)
        return f'<Dataset with {carray} signals and {ctable} tables>'

    @functools.lru_cache(maxsize=MAXSIZE)
    def _load_bcolz(self, name):
        if name in self.carray_names:
            return BcolzSignal(self.base_path / name)
        elif name in self.ctable_names:
            return load_ctable_as_df(self.base_path / name)
        else:
            raise AttributeError(name)


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
