'''
Validates ABR data on a new system

Connect speaker output to EEG input.
'''

import matplotlib.pyplot as plt
import numpy as np

from psi.data.io import abr


def main(filename):
    fh = abr.load(filename)
    epochs = fh.get_epochs()
    epochs_mean = epochs.groupby(['level', 'frequency', 'polarity']).mean()
    levels = epochs_mean.index.unique('level')
    freqs = epochs_mean.index.unique('frequency')
    f1, axes = plt.subplots(len(levels), len(freqs), sharey=True, sharex=True,
                            figsize=(10, 10))
    for l, row in zip(levels, axes):
        for f, ax in zip(freqs, row):
            en = epochs_mean.loc[l, f, -1]
            ax.plot(en, label=f'Neg. pol. average {np.max(np.abs(en)):.2f}')
            ep = epochs_mean.loc[l, f, 1]
            ax.plot(ep, label=f'Pos. pol. average {np.max(np.abs(ep)):.2f}')
            ax.plot(ep+en)

            e_first = epochs.loc[l, f, 1].iloc[0]
            ax.plot(e_first, label='First waveform')
            e_last = epochs.loc[l, f, 1].iloc[-1]
            ax.plot(e_last, label='Last waveform')

            ax.legend()
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlabel('Time (s)')

    f2, axes = plt.subplots(len(levels), len(freqs), sharey=True, sharex=True,
                            figsize=(10, 10))
    for l, row in zip(levels, axes):
        for f, ax in zip(freqs, row):
            e_first = epochs.loc[l, f, 1].iloc[0]
            ax.plot(e_first, label='First waveform')
            e_last = epochs.loc[l, f, 1].iloc[-1]
            ax.plot(e_last, label='Last waveform')
            ax.plot(e_first-e_last, label='Difference')
            ax.legend()
            ax.set_ylabel('Amplitude (V)')
            ax.set_xlabel('Time (s)')

    plt.show()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    main(filename)
