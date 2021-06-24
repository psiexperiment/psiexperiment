'''
Validates ABR data on a new system

Connect speaker output to EEG input.
'''

import matplotlib.pyplot as plt
import numpy as np

from psi.data.io import abr


def main(filename, levels=None, latency=0):
    columns = ['level', 'frequency', 'polarity']
    fh = abr.load(filename)
    epochs = fh.get_epochs(offset=latency, columns=columns)
    epochs_mean = epochs.groupby(columns).mean()
    epochs_size = epochs.groupby(columns).size()

    all_levels = epochs_mean.index.unique('level').tolist()
    all_freqs = epochs_mean.index.unique('frequency').tolist()
    all_levels.sort()
    all_freqs.sort()

    # TODO: Not filtering freqs for now
    freqs = all_freqs[:]

    if not levels:
        levels = all_levels[:]
    levels = sorted(set(levels) & set(all_levels))

    f1, axes = plt.subplots(len(levels), len(freqs), sharey=True, sharex=True,
                            figsize=(10, 10))
    for l, row in zip(levels, axes):
        for f, ax in zip(freqs, row):
            try:
                en = epochs_mean.loc[l, f, -1]
                ax.plot(en, label=f'Neg. pol. average {np.max(np.abs(en)):.2f}')
                e_last = epochs.loc[l, f, 1].iloc[-1]
                ax.plot(e_last, label='Last waveform')
            except KeyError:
                en = None
            try:
                ep = epochs_mean.loc[l, f, 1]
                ax.plot(ep, label=f'Pos. pol. average {np.max(np.abs(ep)):.2f}')
                e_first = epochs.loc[l, f, 1].iloc[0]
                ax.plot(e_first, label='First waveform')
            except KeyError:
                ep = None

            if ep is not None and en is not None:
                ax.plot(ep+en, label='Sum of polarities')

            ax.legend(fontsize=6)
            ax.set_xlabel('Time (s)')

        row[0].set_ylabel(f'Amplitude (V)\n{l:.0f} dB')

    for f, ax in zip(freqs, axes[0]):
        ax.set_title(f'{f:.0f} Hz')

    f2, axes = plt.subplots(len(levels), len(freqs), sharey=True, sharex=True,
                            figsize=(10, 10))
    for l, row in zip(levels, axes):
        for f, ax in zip(freqs, row):
            try:
                e_first = epochs.loc[l, f, 1].iloc[0]
                ax.plot(e_first, label='First waveform')
                e_last = epochs.loc[l, f, 1].iloc[-1]
                ax.plot(e_last, label='Last waveform')
                ax.plot(e_first-e_last, label='Difference')
            except KeyError:
                pass

            ax.legend()
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlabel('Time (s)')

    f3, axes = plt.subplots(len(levels), len(freqs), sharey=True, sharex=True,
                            figsize=(10, 10))
    f4, axes_err = plt.subplots(len(levels), len(freqs), sharey=True,
                                sharex=True, figsize=(10, 10))
    for l, row, row_err in zip(levels, axes, axes_err):
        for f, ax, ax_err in zip(freqs, row, row_err):
            try:
                e = epochs.loc[l, f].loc[:, 1e-3:4e-3]
                e_rms = np.mean(e ** 2, axis=1) ** 0.5
                ax.plot(e_rms.values, label='Tone RMS')

                s = epochs.loc[l, f].loc[:, 5e-3:]
                s_rms = np.mean(s ** 2, axis=1) ** 0.5
                ax.plot(s_rms.values, label='ITI RMS')
                i = s_rms.idxmax()
                w = epochs.loc[l, f].loc[i]
                ax_err.plot(w)
            except KeyError:
                pass

            ax.legend()
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlabel('Time (s)')

    plt.show()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('review-abr-data')
    parser.add_argument('filename')
    parser.add_argument('--levels', nargs='*', type=float)
    parser.add_argument('--latency', default=0.0, type=float)
    args = parser.parse_args()
    print(args)
    main(args.filename, args.levels, args.latency)
