import os.path

import bcolz
import numpy as np


class EpochData:

    def __init__(self, base_path, epoch_name):
        epoch_path = os.path.join(base_path, epoch_name)
        epoch_md_path = os.path.join(base_path,
                                     '{}_metadata'.format(epoch_name))
        self.epoch = bcolz.carray(rootdir=epoch_path)
        self.epoch_md = bcolz.ctable(rootdir=epoch_md_path)

    def get_groups(self, groups):
        return get_epoch_groups(self.epoch, self.epoch_md, groups)


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
