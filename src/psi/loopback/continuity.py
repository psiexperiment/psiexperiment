'''
A pilot signal for finding dropped, repeated or zeroed buffers in a loopback
recording, and the analysis that finds them.

Sound cards don't reliably report under- or overruns (PortAudio's status
flags never fire on the Fireface over ASIO), and a buffer lost between the
stream and psi's stores isn't seen by the sound card at all. The only
dependable evidence is in the recording itself, so `PilotManifest` plays a
known signal on a spare output for the whole run and records it on a spare
input, and `check_continuity` compares the recording with the signal one
short block at a time.

The pilot is broadband noise rather than a tone: a periodic signal's
autocorrelation repeats every period, so a slip of exactly one period would
be invisible and any delay estimate is ambiguous. The noise is a pure
function of the output sample number, so the analysis regenerates it instead
of anything being saved during the run.
'''
from dataclasses import dataclass, field

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence
from scipy import signal


def pseudorandom_sequence(n_samples, n_channels, seed=0, start=0,
                          amplitude=1.0, dtype='float32'):
    '''
    Uniform white noise, statistically independent across channels and
    reproducible from `seed` alone.

    Each channel is drawn from its own bit generator seeded by an
    independently spawned `SeedSequence`, so the channels are not shifts,
    sign flips or decimations of one common sequence.

    `start` selects an offset into each channel's stream without generating
    the samples before it. This relies on `Generator.random()` consuming
    exactly one 64-bit PCG64 output per double, so `PCG64.advance(start)`
    skips precisely `start` samples (an implementation detail of numpy, which
    the tests check).

    Returns an array of shape (n_channels, n_samples), uniform on
    [-amplitude, amplitude).
    '''
    if n_samples < 0:
        raise ValueError('n_samples must be non-negative')
    if start < 0:
        raise ValueError('start must be non-negative')

    y = np.empty((n_channels, n_samples), dtype=dtype)
    for i, ss in enumerate(SeedSequence(seed).spawn(n_channels)):
        bg = PCG64(ss)
        if start:
            bg.advance(start)
        y[i] = 2.0 * Generator(bg).random(n_samples) - 1.0
    y *= amplitude
    return y


class CyclicSequence:
    '''
    A long, band-limited cycle of `pseudorandom_sequence` served as wrapped
    slices, for use directly inside an audio callback.

    One cycle is computed up front, so each block is a copy into the caller's
    buffer: generating noise in the callback would allocate on every block,
    and the resulting churn and GC pauses can cause the very dropouts the
    pilot is meant to find.

    Band-limiting zeroes whole FFT bins, so the cycle stays exactly periodic
    and the wrap has no discontinuity. It keeps a converter's reconstruction
    and anti-alias filters (which strip energy near DC and Nyquist) from
    dominating the residual of an analog loopback.

    The period is long compared with any plausible slip, so correlating
    against the cycle is unambiguous in practice.
    '''

    def __init__(self, n_channels, fs, period=2 ** 20, seed=0, amplitude=1.0,
                 highpass=None, lowpass=None, dtype='float32'):
        self.n_channels = n_channels
        self.fs = fs
        self.period = period
        self.seed = seed
        self.amplitude = amplitude
        self.highpass = highpass
        self.lowpass = lowpass

        y = pseudorandom_sequence(period, n_channels, seed=seed,
                                  amplitude=1.0, dtype='float64')
        if highpass is not None or lowpass is not None:
            freq = np.fft.rfftfreq(period, 1 / fs)
            mask = np.ones(freq.shape, dtype=bool)
            if highpass is not None:
                mask &= (freq >= highpass)
            if lowpass is not None:
                mask &= (freq <= lowpass)
            Y = np.fft.rfft(y, axis=-1)
            Y[:, ~mask] = 0
            y = np.fft.irfft(Y, n=period, axis=-1)

        # Normalize per channel so `amplitude` is the true peak regardless of
        # what band-limiting did to the crest factor.
        y *= amplitude / np.abs(y).max(axis=-1, keepdims=True)
        self.buffer = np.ascontiguousarray(y, dtype=dtype)

    def fill(self, s0, out):
        '''
        Write the `out.shape[-1]` samples starting at absolute sample `s0`
        into `out`, in place. No allocation.
        '''
        n = out.shape[-1]
        if n > self.period:
            raise ValueError('Block is longer than one cycle')
        i = s0 % self.period
        k = min(n, self.period - i)
        out[..., :k] = self.buffer[..., i:i + k]
        if k < n:
            out[..., k:] = self.buffer[..., :n - k]
        return out

    def reference(self, s0, n_samples):
        '''
        The signal spanning [s0, s0 + n_samples) as a new array of shape
        (n_channels, n_samples), wrapping across cycles (`s0` may be
        negative).
        '''
        i = np.arange(s0, s0 + n_samples) % self.period
        return self.buffer[:, i]


def pilot_sequence(fs, seed=0, amplitude=0.1):
    '''
    The single-channel sequence `PilotOutput` plays.
    '''
    return CyclicSequence(1, fs, seed=seed, amplitude=amplitude,
                          highpass=100, lowpass=0.4 * fs)


@dataclass
class Slip:
    #: Time in the recording (s) of the first block that didn't match.
    time: float
    #: How far the recording jumped in the pilot, in samples. Positive means
    #: samples are missing (dropped), negative means samples were repeated.
    #: None if the pilot couldn't be found again afterwards.
    samples: int = None


@dataclass
class Continuity:
    fs: float
    #: Samples per block compared.
    block: int
    #: Offset of the recording into the pilot at the start: recorded sample
    #: k is pilot sample k + offset. 0 if the engine's latency is right.
    offset: int
    #: Median gain from the pilot to the recording.
    gain: float
    #: Start of each block (s) and its residual re the recording (dB).
    times: np.ndarray
    residual_db: np.ndarray
    threshold_db: float
    slips: list = field(default_factory=list)

    @property
    def bad(self):
        return self.residual_db > self.threshold_db

    @property
    def n_bad(self):
        return int(np.sum(self.bad))

    @property
    def bad_times(self):
        return self.times[self.bad]

    @property
    def worst_db(self):
        return float(np.max(self.residual_db)) if len(self.residual_db) \
            else np.nan

    def describe(self):
        if not self.n_bad:
            return 'none'
        times = ', '.join(f'{t:.2f}' for t in self.bad_times[:10])
        more = ', ...' if self.n_bad > 10 else ''
        slips = '; '.join(
            f'{s.time:.2f} s: ' + ('lost' if s.samples is None
                                   else f'{s.samples:+d} samples')
            for s in self.slips[:10])
        return f'{self.n_bad} bad blocks at {times}{more} s' + \
            (f'; slips {slips}' if slips else '')


def _find_offset(x, sequence, s0, guess, max_lag):
    '''
    Offset of `x` (recorded from sample `s0`) into the pilot within
    `max_lag` of `guess`, or None if the pilot isn't clearly in `x`.
    '''
    ref = sequence.reference(s0 + guess - max_lag, len(x) + 2 * max_lag)[0]
    c = signal.correlate(ref, x, 'valid', 'fft')
    j = int(np.argmax(np.abs(c)))
    # Normalized correlation at the peak: ~1 for a clean match, ~1/sqrt(N)
    # by chance.
    seg = ref[j:j + len(x)]
    norm = np.sqrt(np.sum(seg ** 2) * np.sum(x ** 2))
    if norm == 0 or abs(c[j]) / norm < 0.5:
        return None
    return guess + j - max_lag


def check_continuity(x, sequence, block=1024, threshold_db=-40,
                     max_lag=None, window=None):
    '''
    Compare the recording `x` with the pilot `sequence`, one `block` at a
    time.

    The gain is fit per block, so a block is bad only if its shape doesn't
    match: a dropped, repeated or zeroed buffer, or anything else mixed into
    the input. After a bad block the pilot is searched for again, so a slip
    shows up as one bad block and a `Slip` with its size rather than as every
    block after it.

    Parameters
    ----------
    x : array
        Recorded pilot, starting at the engine's first input sample.
    sequence : `CyclicSequence`
        The pilot as played (see `pilot_sequence`).
    block : int
        Samples per block compared.
    threshold_db : float
        Residual (re the recording) above which a block is bad. A digital
        loopback sits far below the default; a misaligned block is near 0 dB.
    max_lag : int
        Largest offset (and slip) to search for, in samples. Defaults to 0.5 s.
    window : int
        Samples used to find the offset. Defaults to 0.1 s.
    '''
    fs = sequence.fs
    x = np.asarray(x, dtype='float64')
    if max_lag is None:
        max_lag = int(0.5 * fs)
    if window is None:
        window = int(0.1 * fs)

    # Find the pilot a little way in, past anything at the very start.
    start = min(len(x) - window, int(0.1 * fs)) if len(x) > window else 0
    offset = _find_offset(x[start:start + window], sequence, start, 0,
                          max_lag)
    if offset is None:
        raise ValueError('The pilot is not in the recording')
    initial_offset = offset

    n_blocks = len(x) // block
    residual = np.empty(n_blocks)
    gains = np.empty(n_blocks)
    slips = []
    for i in range(n_blocks):
        lb = i * block
        rec = x[lb:lb + block]
        ref = sequence.reference(lb + offset, block)[0].astype('float64')
        gain = np.dot(rec, ref) / np.dot(ref, ref)
        error = rec - gain * ref
        power = np.dot(rec, rec)
        # An exact (digital) match leaves no error at all; floor it rather
        # than report -inf.
        residual[i] = 10 * np.log10(max(np.dot(error, error) / power, 1e-30)) \
            if power > 0 else 0
        gains[i] = gain
        if residual[i] > threshold_db:
            # Look for the pilot again after this block.
            ub = lb + block
            new = None
            if ub + window <= len(x):
                new = _find_offset(x[ub:ub + window], sequence, ub, offset,
                                   max_lag)
            if new is not None and new != offset:
                slips.append(Slip(lb / fs, new - offset))
                offset = new
            elif new is None:
                slips.append(Slip(lb / fs, None))

    good = residual <= threshold_db
    gain = float(np.median(gains[good])) if np.any(good) else np.nan
    return Continuity(fs=fs, block=block, offset=initial_offset, gain=gain,
                      times=np.arange(n_blocks) * block / fs,
                      residual_db=residual, threshold_db=threshold_db,
                      slips=slips)
