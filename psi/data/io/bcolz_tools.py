import functools
from glob import glob
import json
import shutil
from pathlib import Path
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

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        carray_names = set(d.parts[-2] for d in self.base_path.glob('*/meta'))
        ctable_names = set(d.parts[-3] for d in self.base_path.glob('*/*/meta'))
        self.carray_names = carray_names
        self.ctable_names = ctable_names

    def __getattr__(self, attr):
        return self._load_bcolz(attr)

    @functools.lru_cache()
    def _load_bcolz(self, name):
        if name in self.carray_names:
            return bcolz.carray(rootdir=self.base_path / name)
        elif name in self.ctable_names:
            return load_ctable_as_df(self.base_path / name)
        else:
            raise AttributeError(name)

    def get_epochs(self, continuous, df, offset=0, duration=None,
                   padding_samples=0, columns=None):
        return get_epochs(
            continuous,
            df,
            offset=offset,
            duration=duration,
            padding_samples=padding_samples,
            columns=columns,
        )


def make_query(trials, base_name='target_tone_'):
    if base_name is not None:
        trials = trials.copy()
        trials = {'{}{}'.format(base_name, k): v for k, v in trials.items()}
    queries = ['({} == {})'.format(k, v) for k, v in trials.items()]
    return ' & '.join(queries)


def format_columns(columns):
    if columns is None:
        columns = ['t0']
        names = ['t0']
    else:
        names = columns + ['t0']
        columns = columns + ['t0']
    return columns, names


def get_epochs(continuous, epoch_md, offset=0, duration=None,
               padding_samples=0, columns=None):

    columns, names = format_columns(columns)
    result_set = epoch_md[columns]
    fs = continuous.attrs['fs']

    epochs = []
    index = []
    max_samples = continuous.shape[-1]

    for i, (_, row) in enumerate(result_set.iterrows()):
        if duration is None:
            duration = row['duration']

        t0 = row['t0']
        lb = int(round((t0+offset)*fs))
        ub = lb + int(round(duration*fs))
        lb -= padding_samples
        ub += padding_samples

        if lb < 0 or ub > max_samples:
            mesg = 'Data missing for epochs {} through {}'
            mesg = mesg.format(i+1, len(epoch_md))
            break

        epoch = continuous[lb:ub]
        epochs.append(epoch[np.newaxis])
        index.append(row)

    n_samples = len(epoch)
    t = (np.arange(n_samples)-padding_samples)/fs + offset
    epochs = np.concatenate(epochs, axis=0)

    index = pd.MultiIndex.from_tuples(index, names=names)
    df = pd.DataFrame(epochs, columns=t, index=index)
    df.sort_index(inplace=True)
    return df


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
