'''
Figures for the run reports that any loopback test can use.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(figure, path):
    figure.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(figure)
    return path.name


def plot_routing(routing, path):
    figure, ax = plt.subplots(figsize=(7, 4.5))
    g = np.abs(routing.gain)
    db = 20 * np.log10(np.maximum(g, 1e-6))
    # Below the threshold is just how well independent noise correlates by
    # chance, not a route.
    shown = np.where(g > routing.threshold, db, np.nan)
    im = ax.imshow(shown, vmin=-30, vmax=0, cmap='viridis')
    ax.set_xticks(range(len(routing.output_names)), routing.output_names,
                  rotation=45, ha='right')
    ax.set_yticks(range(len(routing.input_names)), routing.input_names)
    for i in range(db.shape[0]):
        for o in range(db.shape[1]):
            if g[i, o] > routing.threshold:
                ax.text(o, i, f'{db[i, o]:.1f}', ha='center', va='center',
                        color='k', fontsize=8)
    ax.set_xlabel('Output')
    ax.set_ylabel('Input')
    figure.colorbar(im, label='Gain (dB)')
    ax.set_title(f'Routing at {routing.fs / 1e3:.0f} kHz')
    return save(figure, path)


def plot_timing(fit, path, expected_ioi=None):
    '''
    Lag of each stimulus re the common latency, and the interval between
    onsets. `expected_ioi` (seconds, scalar or list) are the intervals the
    paradigm should produce; each interval is plotted as its error from the
    nearest one, in samples. Without it, the intervals themselves are
    plotted (e.g., with rate jitter).
    '''
    fs = fit.fs
    e = fit.epochs
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    ax.plot(e.onset, e.lag, 'o', ms=3)
    ax.set_xlabel('Onset (s)')
    ax.set_ylabel('Lag re latency (samples)')
    ax.set_title(f'Onset jitter (latency {fit.latency} samples)')
    lim = max(2, np.nanmax(np.abs(e.lag)) + 1)
    ax.set_ylim(-lim, lim)

    ax = axes[1]
    ioi = np.diff(e.onset.values)
    if expected_ioi is None:
        ax.plot(e.onset.values[1:], ioi * 1e3, 'o', ms=3)
        ax.set_ylabel('Interval (ms)')
        ax.set_title('Interval from previous onset')
    else:
        expected = np.atleast_1d(expected_ioi)
        nearest = expected[np.argmin(np.abs(ioi[:, None] - expected), axis=1)]
        error = (ioi - nearest) * fs
        ax.plot(e.onset.values[1:], error, 'o', ms=3)
        ax.set_ylabel('Error (samples)')
        label = ', '.join(f'{i * 1e3:.3f}' for i in expected)
        ax.set_title(f'Interval from previous onset, re {label} ms')
        lim = max(2, np.nanmax(np.abs(error)) + 1)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel('Onset (s)')
    figure.tight_layout()
    return save(figure, path)


def plot_continuity(continuity, path):
    '''
    Residual of every block of the pilot over the run, with the threshold
    for a bad block and where the pilot slipped.
    '''
    c = continuity
    figure, ax = plt.subplots(figsize=(10, 3))
    ax.plot(c.times, c.residual_db, 'k', lw=0.5)
    ax.axhline(c.threshold_db, color='C3', ls=':', lw=1,
               label='Bad block threshold')
    if c.n_bad:
        ax.plot(c.bad_times, c.residual_db[c.bad], 'o', color='C3', ms=4,
                label='Bad block')
    for i, slip in enumerate(c.slips):
        ax.axvline(slip.time, color='C1', lw=1,
                   label='Slip' if i == 0 else None)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Residual re recording (dB)')
    ax.set_title(f'Pilot: {c.n_bad} bad blocks of {len(c.times)} '
                 f'({c.block} samples each), offset {c.offset} samples')
    ax.legend(loc='upper right', fontsize=8)
    return save(figure, path)
