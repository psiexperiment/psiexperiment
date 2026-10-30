'''
Compare what a paradigm recorded through the loopback with what it says it
played.

Each stimulus the paradigm logs (its onset plus whatever is needed to
regenerate it) is regenerated and fit to the recording. A digital loopback is
exact, so after allowing for the sound card's round-trip delay and the
route's gain, every stimulus should match its template to well below the
noise floor, start at the same delay after its logged onset, and nothing
should be in the recording that wasn't logged.
'''
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal


@dataclass
class Fit:
    #: Delay from logged onset to the stimulus in the recording, in samples,
    #: common to every stimulus.
    latency: int
    #: One row per stimulus: `lag` (samples, relative to `latency`),
    #: `gain` (recorded / template; negative if the polarity is inverted),
    #: `residual_db` (what the template leaves unexplained, re the stimulus)
    #: and `onset` (seconds, in the recording).
    epochs: pd.DataFrame
    #: The recording, and the sum of every fitted template.
    recorded: np.ndarray
    reconstructed: np.ndarray
    fs: float

    @property
    def unexplained_db(self):
        '''
        Energy in the recording that the fitted stimuli don't account for
        (e.g., stimuli that were played but not logged), re the recording.
        '''
        error = self.recorded - self.reconstructed
        return 10 * np.log10(np.sum(error ** 2) / np.sum(self.recorded ** 2))


def find_latency(x, fs, onsets, templates, max_latency=0.5):
    '''
    Delay, in samples, from the logged onsets to where the stimuli appear in
    `x`.

    Found by correlating the recording against all the templates placed at
    their logged onsets, which a single periodic stimulus could not pin down
    but the onset of the whole train does. The search starts at the first
    logged onset, so a delay longer than the silence before the train is
    still found.
    '''
    max_lag = int(max_latency * fs)
    expected = np.zeros(len(x) + max_lag)
    for t0, template in zip(onsets, templates):
        i = int(round(t0 * fs))
        # The recording may stop partway through the last stimulus.
        n = max(0, min(len(template), len(expected) - i))
        expected[i:i + n] += template[:n]
    expected = expected[:len(x)]
    c = signal.correlate(x, expected, 'full', 'fft')
    zero = len(expected) - 1
    c = c[zero:zero + max_lag]
    return int(np.argmax(c))


def fit_stimuli(x, fs, onsets, templates, latency=None, search=0.002):
    '''
    Fit each template to the recording near its logged onset.

    Parameters
    ----------
    x : array
        The recording.
    onsets : array
        Logged onset of each stimulus, in seconds.
    templates : list of arrays
        What each stimulus should be.
    latency : int or None
        Delay in samples; found with `find_latency` if None.
    search : float
        How far either side of the expected position to look, in seconds.
    '''
    if latency is None:
        latency = find_latency(x, fs, onsets, templates)
    n_search = int(search * fs)
    reconstructed = np.zeros_like(x)
    rows = []
    for t0, template in zip(onsets, templates):
        i = int(round(t0 * fs)) + latency
        lb = i - n_search
        ub = i + len(template) + n_search
        if lb < 0 or ub > len(x) or not np.any(template):
            # Off the ends of the recording, or nothing to find (e.g., a
            # silent reference).
            rows.append({'lag': np.nan, 'gain': np.nan, 'residual_db': np.nan,
                         'onset': np.nan})
            continue
        segment = x[lb:ub]
        c = signal.correlate(segment, template, 'valid')
        k = int(np.argmax(np.abs(c)))
        s = x[lb + k:lb + k + len(template)]
        gain = np.dot(s, template) / np.dot(template, template)
        error = s - gain * template
        with np.errstate(divide='ignore'):
            residual_db = 10 * np.log10(np.sum(error ** 2) / np.sum(s ** 2))
        reconstructed[lb + k:lb + k + len(template)] += gain * template
        rows.append({
            'lag': k - n_search,
            'gain': gain,
            'residual_db': residual_db,
            'onset': (lb + k) / fs,
        })
    return Fit(latency, pd.DataFrame(rows), x, reconstructed, fs)


def gain_db(gain):
    return 20 * np.log10(np.abs(gain))


def db_ratio(a, b):
    '''
    RMS of `a` re RMS of `b`, in dB (along the last axis).
    '''
    rms_a = np.sqrt(np.mean(np.square(a), axis=-1))
    rms_b = np.sqrt(np.mean(np.square(b), axis=-1))
    with np.errstate(divide='ignore'):
        return 20 * np.log10(rms_a / rms_b)
