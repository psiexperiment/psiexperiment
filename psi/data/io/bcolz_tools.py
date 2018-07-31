from glob import glob
import json
import shutil
import os.path

import bcolz
import numpy as np
import pandas as pd


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


class Dataset:

    def __init__(self, base_path, continuous_names=None, epoch_names=None):
        self.continuous = {}
        if continuous_names is not None:
            for name in continuous_names:
                path = os.path.join(base_path, name)
                self.continuous[name] = bcolz.carray(rootdir=path)

        self.epoch = {}
        self.epoch_md = {}
        if epoch_names is not None:
            for name in epoch_names:
                # Open the epoch
                path = os.path.join(base_path, name)
                self.epoch[name] = bcolz.carray(rootdir=path)
                # Load the metadata
                path = os.path.join(base_path, f'{name}_metadata')
                md = bcolz.ctable(rootdir=path)
                self.epoch_md[name] = md.todataframe()

    def extract_epochs_df(self, signal_name, epoch_name, columns=None,
                          offset=0, duration=None, padding_samples=0):
        return get_epochs_df(
            self.continuous[signal_name],
            self.epoch_md[epoch_name],
            columns=columns,
            offset=offset,
            duration=duration,
            padding_samples=padding_samples
        )


def get_epochs_df(continuous, epoch_md, columns, offset=0, duration=None,
                  padding_samples=0):

    fs = continuous.attrs['fs']
    if columns is None:
        columns = []
    columns = list(columns) + ['t0']
    keys = []
    data = []
    max_samples = continuous.shape[-1]

    for _, row in epoch_md.iterrows():
        if duration is None:
            duration = row['duration']
        key = tuple(row[c] for c in columns)
        keys.append(key)
        t0 = row['t0']
        lb = round((t0+offset)*fs)
        ub = lb + round(duration*fs)
        lb -= padding_samples
        ub += padding_samples

        d = continuous[lb:ub]

        if lb < 0 or ub > max_samples:
            lb_pad = max(0-lb, 0)
            ub_pad = max(ub-max_samples, 0)
            d = np.pad(d, (lb_pad, ub_pad), mode='constant',
                       constant_values=np.nan)

        t = pd.Index(np.arange((ub-lb))/fs, name='time')
        d = pd.Series(d, index=t, name='signal')
        data.append(d)

    return pd.concat(data, keys=keys, names=columns).unstack('time')


def get_epoch_groups(epoch, epoch_md, groups):
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
