import numpy as np
import pandas as pd


def get_epoch_groups(fs, groups, metadata, array):
    df = pd.DataFrame(metadata)

    duration = df['duration'].values
    samples = np.round(duration * fs)
    offset = samples.cumsum() - samples[0]
    df['samples'] = samples
    df['offset'] = offset

    epochs = {}
    for keys, g_df in df.groupby(groups):
        data = []
        for _, row in g_df.iterrows():
            o = int(row['offset'])
            s = int(row['samples'])
            d = array[o:o+s][np.newaxis]
            data.append(d)
        epochs[keys] = np.concatenate(data, axis=0)
    return epochs
