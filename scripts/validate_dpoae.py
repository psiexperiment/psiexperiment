import matplotlib.pyplot as plt

from psi.data.io import dpoae


def main(filename):
    fh = dpoae.load(filename)
    data = fh.results[['f1_frequency', 'f1_level', 'meas_f1_level']]
    freqs = data['f1_frequency'].unique()

    figure, axes = plt.subplots(1, len(freqs), sharex=True, sharey=True, figsize=(10, 3))
    for freq, ax in zip(freqs, axes):
        d = data.query(f'f1_frequency == {freq}')
        ax.plot(d['f1_level'], d['meas_f1_level'], 'ko-')
        ax.set_title(f'F1 frequency {freq}')
        ax.set_xlabel('Requested level (dB gain)')
        ax.set_ylabel('Measured level (dB gain)')

    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    main(filename)
