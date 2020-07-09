import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import signal
from fractions import gcd

from psi.util import as_numeric


def db(target, reference=1):
    target = as_numeric(target)
    reference = as_numeric(reference)
    return 20*np.log10(target/reference)


def dbi(db, reference=1):
    db = as_numeric(db)
    return (10**(db/20))*reference


def dbtopa(db):
    '''
    Convert dB SPL to Pascal

    .. math:: 10^(dB/20.0)/20e-6

    >>> print dbtopa(100)
    2.0
    >>> print dbtopa(120)
    20.0
    >>> print patodb(dbtopa(94.0))
    94.0

    Will also take sequences:
    >>> print dbtopa([80, 100, 120])
    [  0.2   2.   20. ]
    '''
    return dbi(db, 20e-6)


def patodb(pa):
    '''
    Convert Pascal to dB SPL

    .. math:: 20*log10(pa/20e-6)

    >>> print round(patodb(1))
    94.0
    >>> print patodb(2)
    100.0
    >>> print patodb(0.2)
    80.0

    Will also take sequences:
    >>> print patodb([0.2, 2.0, 20.0])
    [  80.  100.  120.]
    '''
    return db(pa, 20e-6)


def normalize_rms(waveform, out=None):
    '''
    Normalize RMS power to 1 (typically used when generating a noise waveform
    that will be scaled by a calibration factor)

    waveform : array_like
        Input array.
    out : array_like
        An array to store the output.  Must be the same shape as `waveform`.
    '''
    return np.divide(waveform, rms(waveform), out)


def csd(s, fs, window=None, waveform_averages=None):
    if waveform_averages is not None:
        new_shape = (waveform_averages, -1) + s.shape[1:]
        s = s.reshape(new_shape).mean(axis=0)
    s = signal.detrend(s, type='linear', axis=-1)
    n = s.shape[-1]
    if window is not None:
        w = signal.get_window(window, n)
        s = w/w.mean()*s
    return np.fft.rfft(s, axis=-1)/n

def phase(s, fs, window=None, waveform_averages=None, unwrap=True):
    c = csd(s, fs, window, waveform_averages)
    p = np.angle(c)
    if unwrap:
        p = np.unwrap(p)
    return p


def psd(s, fs, window=None, waveform_averages=None):
    c = csd(s, fs, window, waveform_averages)
    return 2*np.abs(c)/np.sqrt(2.0)


def psd_freq(s, fs):
    return np.fft.rfftfreq(s.shape[-1], 1.0/fs)


def psd_df(s, fs, *args, **kw):
    p = psd(s, fs)
    freqs = pd.Index(psd_freq(s, fs), name='frequency')
    if p.ndim == 1:
        name = s.name if isinstance(s, pd.Series) else 'psd'
        return pd.Series(p, index=freqs, name=name)
    else:
        index = s.index if isinstance(s, pd.DataFrame) else None
        return pd.DataFrame(p, columns=freqs, index=index)


def tone_conv(s, fs, frequency, window=None):
    frequency_shape = tuple([Ellipsis] + [np.newaxis]*s.ndim)
    frequency = np.asarray(frequency)[frequency_shape]
    s = signal.detrend(s, type='linear', axis=-1)
    n = s.shape[-1]
    if window is not None:
        w = signal.get_window(window, n)
        s = w/w.mean()*s
    t = np.arange(n)/fs
    r = 2.0*s*np.exp(-1.0j*(2.0*np.pi*t*frequency))
    return np.mean(r, axis=-1)


def tone_power_conv(s, fs, frequency, window=None):
    r = tone_conv(s, fs, frequency, window)
    return np.abs(r)/np.sqrt(2.0)


def tone_phase_conv(s, fs, frequency, window=None):
    r = tone_conv(s, fs, frequency, window)
    return np.angle(r)


def tone_power_fft(s, fs, frequency, window=None):
    power = psd(s, fs, window)
    freqs = psd_freq(s, fs)
    flb, fub = freqs*0.9, freqs*1.1
    mask = (freqs >= flb) & (freqs < fub)
    return power[..., mask].max(axis=-1)


def tone_phase_fft(s, fs, frequency, window=None):
    p = phase(s, fs, window, unwrap=False)
    freqs = psd_freq(s, fs)
    flb, fub = freqs*0.9, freqs*1.1
    mask = (freqs >= flb) & (freqs < fub)
    return p[..., mask].max(axis=-1)


def tone_power_conv_nf(s, fs, frequency, window=None):
    samples = s.shape[-1]
    resolution = fs/samples
    frequencies = frequency+np.arange(-2, 3)*resolution
    magnitude = tone_power_conv(s, fs, frequencies, window)
    nf_rms = magnitude[(0, 1, 3, 4), ...].mean(axis=0)
    tone_rms = magnitude[2]
    return nf_rms, tone_rms


def analyze_mic_sens(ref_waveforms, exp_waveforms, vrms, ref_mic_gain,
                     exp_mic_gain, output_gain, ref_mic_sens, **kwargs):

    ref_data = analyze_tone(ref_waveforms, mic_gain=ref_mic_gain, **kwargs)
    exp_data = analyze_tone(exp_waveforms, mic_gain=exp_mic_gain, **kwargs)

    # Actual output SPL
    output_spl = ref_data['mic_rms']-ref_mic_sens-db(20e-6)
    # Output SPL assuming 0 dB gain and 1 VRMS
    norm_output_spl = output_spl-output_gain-db(vrms)
    # Exp mic sensitivity in dB(V/Pa)
    exp_mic_sens = exp_data['mic_rms']+ref_mic_sens-ref_data['mic_rms']

    result = {
        'output_spl': output_spl,
        'norm_output_spl': norm_output_spl,
        'exp_mic_sens': exp_mic_sens,
        'output_gain': output_gain,
    }
    shared = ('time', 'frequency')
    result.update({k: ref_data[k] for k in shared})
    t = {'ref_'+k: ref_data[k] for k, v in ref_data.items() if k not in shared}
    result.update(t)
    t = {'exp_'+k: exp_data[k] for k, v in exp_data.items() if k not in shared}
    result.update(t)
    return result


def thd(s, fs, frequency, harmonics=3, window=None):
    ph = np.array([tone_power_conv(s, fs, frequency*(i+1), window)[np.newaxis] \
                   for i in range(harmonics)])
    ph = np.concatenate(ph, axis=0)
    return (np.sum(ph[1:]**2, axis=0)**0.5)/ph[0]


def analyze_tone(waveforms, frequency, fs, mic_gain, trim=0, thd_harmonics=3):
    trim_n = int(trim*fs)
    waveforms = waveforms[:, trim_n:-trim_n]

    # Get average tone power across channels
    power = tone_power_conv(waveforms, fs, frequency, window='flattop')
    power = db(power).mean(axis=0)

    average_waveform = waveforms.mean(axis=0)
    time = np.arange(len(average_waveform))/fs

    # Correct for gains (i.e. we want to know the *actual* Vrms at 0 dB input
    # and 0 dB output gain).
    power -= mic_gain

    #max_harmonic = np.min(int(np.floor((fs/2.0)/frequency)), thd_harmonics)
    harmonics = []
    for i in range(thd_harmonics):
        f_harmonic = frequency*(i+1)
        p = tone_power_conv(waveforms, fs, f_harmonic, window='flattop')
        p_harmonic = db(p).mean(axis=0)
        harmonics.append({
            'harmonic': i+1,
            'frequency': f_harmonic,
            'mic_rms': p_harmonic,
        })

    harmonic_v = []
    for h_info in harmonics:
        harmonic_v.append(dbi(h_info['mic_rms']))
    harmonic_v = np.asarray(harmonic_v)[:thd_harmonics]
    thd = (np.sum(harmonic_v[1:]**2)**0.5)/harmonic_v[0]

    return {
        'frequency': frequency,
        'time': time,
        'mic_rms': power,
        'thd': thd,
        'mic_waveform': average_waveform,
        'harmonics': harmonics,
    }


def rms(s, detrend=False):
    if detrend:
        s = signal.detrend(s, axis=-1)
    return np.mean(s**2, axis=-1)**0.5


def golay_pair(n=15):
    '''
    Generate pair of Golay sequences
    '''
    a0 = np.array([1, 1])
    b0 = np.array([1, -1])
    for i in range(n):
        a = np.concatenate([a0, b0])
        b = np.concatenate([a0, -b0])
        a0, b0 = a, b
    return a.astype(np.float32), b.astype(np.float32)


def transfer_function(stimulus, response, fs):
    response = response[:len(stimulus)]
    h_response = np.fft.rfft(response, axis=-1)
    h_stimulus = np.fft.rfft(stimulus, axis=-1)
    freq = psd_freq(response, fs)
    return freq, 2*np.abs(h_response*np.conj(h_stimulus))


def golay_tf(a, b, a_signal, b_signal, fs):
    '''
    Estimate system transfer function from Golay sequence

    Implements algorithm as described in Zhou et al. 1992.
    '''
    a_signal = a_signal[..., :len(a)]
    b_signal = b_signal[..., :len(b)]
    ah_psd = np.fft.rfft(a_signal, axis=-1)
    bh_psd = np.fft.rfft(b_signal, axis=-1)
    a_psd = np.fft.rfft(a)
    b_psd = np.fft.rfft(b)
    h_omega = (ah_psd*np.conj(a_psd) + bh_psd*np.conj(b_psd))/(2*len(a))
    freq = psd_freq(a, fs)
    h_psd = np.abs(h_omega)
    h_phase = np.unwrap(np.angle(h_omega))
    return freq, h_psd, h_phase


def golay_ir(n, a, b, a_signal, b_signal):
    '''
    Estimate system impulse response from Golay sequence

    Implements algorithm described in Zhou et al. 1992
    '''
    a_signal = a_signal.mean(axis=0)
    b_signal = b_signal.mean(axis=0)
    a_conv = np.apply_along_axis(np.convolve, 1, a_signal, a[::-1], 'full')
    b_conv = np.apply_along_axis(np.convolve, 1, b_signal, b[::-1], 'full')
    return 1.0/(2.0*n)*(a_conv+b_conv)[..., -len(a):]


def summarize_golay(fs, a, b, a_response, b_response, waveform_averages=None):

    if waveform_averages is not None:
        n_epochs, n_time = a_response.shape
        new_shape = (waveform_averages, -1, n_time)
        a_response = a_response.reshape(new_shape).mean(axis=0)
        b_response = b_response.reshape(new_shape).mean(axis=0)

    time = np.arange(a_response.shape[-1])/fs
    freq, tf_psd, tf_phase = golay_tf(a, b, a_response, b_response, fs)
    tf_psd = tf_psd.mean(axis=0)
    tf_phase = tf_phase.mean(axis=0)

    return {
        'psd': tf_psd,
        'phase': tf_phase,
        'frequency': freq,
    }


def freq_smooth(frequency, power, bandwidth=20):
    '''
    Uses Konno & Ohmachi (1998) algorithm
    '''
    smoothed = []
    old = np.seterr(all='ignore')
    for f in frequency:
        if f == 0:
            # Special case for divide by 0
            k = np.zeros_like(frequency)
        else:
            r = bandwidth*np.log10(frequency/f)
            k = (np.sin(r)/r)**4
            # Special case for np.log10(0/frequency)
            k[0] = 0
            # Special case where ratio is 1 (log of ratio is set to 0)
            k[frequency == f] = 1
            # Equalize weights
            k /= k.sum(axis=0)
        smoothed.append(np.sum(power*k))
    np.seterr(**old)
    return np.array(smoothed)


def ir_iir(impulse_response, fs, smooth=None, *args, **kwargs):
    csd = np.fft.rfft(impulse_response)
    psd = np.abs(csd)/len(impulse_response)
    phase = np.unwrap(np.angle(csd))
    frequency = np.fft.rfftfreq(len(impulse_response), fs**-1)

    # Smooth in the frequency domain
    if smooth is not None:
        psd = dbi(freq_smooth(frequency, db(psd), smooth))
        phase = freq_smooth(frequency, phase, smooth)

    return iir(psd, phase, frequency, *args, **kwargs)


def iir(psd, phase, frequency, cutoff=None, phase_correction=None,
        truncate=None, truncate_spectrum=False, reference='mean'):
    '''
    Given the impulse response, compute the inverse impulse response.

    Parameters
    ----------
    # TODO

    Note
    ----
    Specifying the cutoff range is highly recommended to get a well-behaved
    function.
    '''
    # Equalize only a subset of the calibrated frequencies
    if cutoff is not None:
        lb, ub = cutoff
        m = (frequency >= lb) & (frequency < ub)
        inverse_psd = psd[m].mean()/psd
        inverse_psd[~m] = 1
    else:
        inverse_psd = psd.mean()/psd

    if phase_correction == 'linear':
        m, b = np.polyfit(frequency, phase, 1)
        inverse_phase = 2*np.pi*np.arange(len(frequency))*m+b
    elif phase_correction == 'subtract':
        inverse_phase = 2*np.pi-phase
    else:
        inverse_phase = phase

    filtered_spectrum = inverse_psd*np.exp(inverse_phase*1j)

    if truncate_spectrum:
        orig_ub = np.round(frequency[-1])
        ub = np.round(ub)
        filtered_spectrum = filtered_spectrum[frequency <= ub]
        iir = truncated_ifft(filtered_spectrum, orig_ub, ub)
    else:
        iir = np.fft.irfft(filtered_spectrum)

    if truncate:
        truncate_samples = int(truncate*fs)
        iir = iir[:truncate_samples]

    return iir


def truncated_ifft(spectrum, original_fs, truncated_fs):
    iir = np.fft.irfft(spectrum)
    lcm = original_fs*truncated_fs/gcd(original_fs, truncated_fs)
    up = lcm/truncated_fs
    down = lcm/original_fs
    iir = signal.resample_poly(iir, up, down)
    iir *= truncated_fs/original_fs
    return iir


def save_calibration(channels, filename):
    from psi.util import get_tagged_values
    from json_tricks import dump
    settings = {}
    for channel in channels:
        metadata = get_tagged_values(channel.calibration, 'metadata')
        metadata['calibration_type'] = channel.calibration.__class__.__name__
        if 'source' in metadata:
            metadata['source'] = str(metadata['source'])
        settings[channel.name] = metadata
    with open(filename, 'w') as fh:
        dump(settings, fh, indent=4)


def load_calibration(filename, channels):
    '''
    Load calibration configuration for hardware from json file
    '''
    from json_tricks import load
    from psi.controller.calibration.api import calibration_registry
    with open(filename, 'r') as fh:
        settings = load(fh)
    channels = {c.name: c for c in channels}
    for c_name, c_calibration in settings.items():
        log.debug('Loading calibration %s with data %r', c_name, c_calibration)
        channels[c_name].calibration = \
            calibration_registry.from_dict(**c_calibration)
