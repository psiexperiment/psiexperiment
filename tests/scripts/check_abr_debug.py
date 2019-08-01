# To use this script, connect a cable from the speaker output to the EEG input
# on your data acquisition card. You will need to ensure that the ABR is
# configured so that there is no filtering and in-ear calibration is not
# applied. You will likely want to set the level of the ABR tone to 0, -6 and
# -12 dB (i.e., these are attenuation values relative to 1 dB Vrms).


def main(pathname):
    import matplotlib.pyplot as plt
    from psi.data.io import abr
    fh = abr.ABRFile(pathname)
    epochs = fh.get_epochs()
    mean_waveforms = epochs.groupby(['frequency', 'level', 'polarity']).mean()
    n_freqs = len(mean_waveforms.index.unique(level='frequency'))
    n_levels = len(mean_waveforms.index.unique(level='level'))
    figure, axes = plt.subplots(n_freqs, n_levels, 
                                figsize=(n_levels*2, n_freqs*2), 
                                sharex=True, sharey=True)

    grouping = mean_waveforms.groupby(['frequency', 'level'])
    for ax, (_, w) in zip(axes.ravel(), grouping):
        ax.plot(w.T, 'k-')

    #mean_waveforms.T.plot()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check ABR debugging')
    parser.add_argument('pathname', type=str, help='Filename containing ABR data')
    args = parser.parse_args()
    main(args.pathname)
