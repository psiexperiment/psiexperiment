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
