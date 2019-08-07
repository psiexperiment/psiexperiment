import logging
log = logging.getLogger(__name__)

from functools import partial
import time

import numpy as np
import pandas as pd

from .util import tone_power_conv, tone_phase_conv, db
from .calibration import (FlatCalibration, PointCalibration,
                          CalibrationTHDError, CalibrationNFError)


from psi.token.primitives import ToneFactory, SilenceFactory


def process_tone(fs, signal, frequency, min_snr=None, max_thd=None,
                 thd_harmonics=3, silence=None):
    '''
    Compute the RMS at the specified frequency. Check for distortion.

    Parameters
    ----------
    fs : float
        Sampling frequency of signal
    signal : ndarray
        Last dimension must be time. If more than one dimension, first
        dimension must be repetition.
    frequency : float
        Frequency (Hz) to analyze
    min_snr : {None, float}
        If specified, must provide a noise floor measure (silence). The ratio,
        in dB, of signal RMS to silence RMS must be greater than min_snr. If
        not, a CalibrationNFError is raised.
    max_thd : {None, float}
        If specified, ensures that the total harmonic distortion, as a
        percentage, is less than max_thd. If not, a CalibrationTHDError is
        raised.
    thd_harmonics : int
        Number of harmonics to compute. If you pick too many, some harmonics
        may be above the Nyquist frequency and you'll get an exception.
    thd_harmonics : int
        Number of harmonics to compute. If you pick too many, some harmonics
        may be above the Nyquist frequency and you'll get an exception.
    silence : {None, ndarray}
        Noise floor measurement. Required for min_snr. Shape must match signal
        in all dimensions except the first and last.

    Returns
    -------
    result : pandas Series
        Series containing rms, snr, thd and frequency.
    '''
    harmonics = frequency * (np.arange(thd_harmonics) + 1)

    # This returns an array of [harmonic, repetition, channel]. Here, rms[0] is
    # the rms at the fundamental frequency. rms[1:] is the rms at all the
    # harmonics.
    signal = np.atleast_2d(signal)
    rms = tone_power_conv(signal, fs, harmonics, 'flattop')
    phase = tone_phase_conv(signal, fs, frequency, 'flattop')

    # Compute the mean RMS at F0 across all repetitions
    rms = rms.mean(axis=1)
    freq_rms = rms[0]

    freq_phase = phase.mean(axis=0)
    freq_phase_deg = np.rad2deg(freq_phase)

    # Compute the harmonic distortion as a percent
    thd = np.sqrt(np.sum(rms[1:]**2))/freq_rms * 100

    # If a silent period has been provided, use this to estimat the signal to
    # noise ratio. As an alternative, could we just use the "sidebands"?
    if silence is not None:
        silence = np.atleast_2d(silence)
        noise_rms = tone_power_conv(silence, fs, frequency, 'flattop')
        noise_rms = noise_rms.mean(axis=0)
        freq_snr = db(freq_rms, noise_rms)
        if min_snr is not None:
            if np.any(freq_snr < min_snr):
                raise CalibrationNFError(frequency, freq_snr)
    else:
        freq_snr = np.full_like(freq_rms, np.nan)

    if max_thd is not None and np.any(thd > max_thd):
        raise CalibrationTHDError(frequency, thd)

    # Concatenate and return as a record array
    result = np.concatenate((freq_rms[np.newaxis], freq_snr[np.newaxis],
                             thd[np.newaxis]))

    data = {'rms': freq_rms, 'snr': freq_snr, 'thd': thd,
            'phase': freq_phase, 'phase_degrees': freq_phase_deg}

    if result.ndim == 1:
        return pd.Series(data)
    else:
        return pd.DataFrame(data)


def tone_power(engine, frequencies, ao_channel_name, ai_channel_names, gains=0,
               vrms=1, repetitions=2, min_snr=None, max_thd=None, thd_harmonics=3,
               duration=0.1, trim=0.01, iti=0.01, debug=False):
    '''
    Given a single output, measure response in multiple input channels.

    Parameters
    ----------
    TODO

    Returns
    -------
    result : pandas DataFrame
        Dataframe will be indexed by output channel name and frequency. Columns
        will be rms (in V), snr (in DB) and thd (in percent).
    '''
    if not isinstance(ao_channel_name, str):
        raise ValueError('Can only specify one output channel')

    from psi.controller.api import ExtractEpochs, FIFOSignalQueue

    frequencies = np.asarray(frequencies)
    calibration = FlatCalibration.as_attenuation(vrms=vrms)

    # Create a copy of the engine containing only the channels required for
    # calibration.
    channel_names = ai_channel_names + [ao_channel_name]
    cal_engine = engine.clone(channel_names)
    ao_channel = cal_engine.get_channel(ao_channel_name)
    ai_channels = [cal_engine.get_channel(name) for name in ai_channel_names]

    ao_fs = ao_channel.fs
    ai_fs = ai_channels[0].fs

    # Ensure that input channels are synced to the output channel 
    device_name = ao_channel.device_name
    ao_channel.start_trigger = ''
    for channel in ai_channels:
        channel.start_trigger = f'/{device_name}/ao/StartTrigger'

    samples = int(ao_fs*duration)

    if np.isscalar(gains):
        gains = [gains] * len(frequencies)

    # Build the signal queue
    queue = FIFOSignalQueue()
    queue.set_fs(ao_fs)
    for frequency, gain in zip(frequencies, gains):
        factory = ToneFactory(ao_fs, gain, frequency, 0, 1, calibration)
        waveform = factory.next(samples)
        md = {'gain': gain, 'frequency': frequency}
        queue.append(waveform, repetitions, iti, metadata=md)

    factory = SilenceFactory(ao_fs, calibration)
    waveform = factory.next(samples)
    md = {'gain': -400, 'frequency': 0}
    queue.append(waveform, repetitions, iti, metadata=md)

    # Add the queue to the output channel
    output = ao_channel.add_queued_epoch_output(queue, auto_decrement=True)

    # Activate the output so it begins as soon as acquisition begins
    output.activate(0)

    # Create a dictionary of lists. Each list maps to an individual input
    # channel and will be used to accumulate the epochs for that channel.
    data = {ai_channel.name: [] for ai_channel in ai_channels}
    samples = {ai_channel.name: [] for ai_channel in ai_channels}

    def accumulate(epochs, epoch):
        epochs.extend(epoch)

    for ai_channel in ai_channels:
        cb = partial(accumulate, data[ai_channel.name])
        epoch_input = ExtractEpochs(epoch_size=duration)
        queue.connect(epoch_input.queue.append)
        epoch_input.add_callback(cb)
        ai_channel.add_input(epoch_input)
        ai_channel.add_callback(samples[ai_channel.name].append)

    cal_engine.start()
    while not epoch_input.complete:
        time.sleep(0.1)
    cal_engine.stop()

    result = []
    for ai_channel in ai_channels:
        # Process data from channel
        epochs = [epoch['signal'][np.newaxis] for epoch in data[ai_channel.name]]
        signal = np.concatenate(epochs)
        signal.shape = [-1, repetitions] + list(signal.shape[1:])

        if trim != 0:
            trim_samples = round(ai_channel.fs * trim)
            signal = signal[..., trim_samples:-trim_samples]

        # Loop through each frequency (silence will always be the last one)
        silence = signal[-1]
        signal = signal[:-1]
        channel_result = []
        for f, s in zip(frequencies, signal):
            f_result = process_tone(ai_channel.fs, s, f, min_snr, max_thd,
                                    thd_harmonics, silence)
            f_result['frequency'] = f
            if debug:
                f_result['waveform'] = s
            channel_result.append(f_result)

        df = pd.DataFrame(channel_result)
        df['channel_name'] = ai_channel.name
        result.append(df)

    return pd.concat(result).set_index(['channel_name', 'frequency'])


def tone_spl(engine, *args, **kwargs):
    '''
    Given a single output, measure resulting SPL in multiple input channels.

    Parameters
    ----------
    TODO

    Returns
    -------
    result : pandas DataFrame
        Dataframe will be indexed by output channel name and frequency. Columns
        will be rms (in V), snr (in DB), thd (in percent) and spl (measured dB
        SPL according to the input calibration).
    '''
    result = tone_power(engine, *args, **kwargs)

    def map_spl(series, engine):
        channel_name, frequency = series.name
        channel = engine.get_channel(channel_name)
        spl = channel.calibration.get_spl(frequency, series['rms'])
        series['spl'] = spl
        return series

    return result.apply(map_spl, axis=1, args=(engine,))


def tone_sens(engine, frequencies, gain=-40, vrms=1, *args, **kwargs):
    '''
    Given a single output, measure sensitivity of output based on multiple
    input channels.

    Parameters
    ----------
    TODO

    Returns
    -------
    result : pandas DataFrame
        Dataframe will be indexed by output channel name and frequency. Columns
        will be rms (in V), snr (in DB), thd (in percent), spl (measured dB
        SPL according to the input calibration) norm_spl (the output, in dB
        SPL, that would be generated assuming the tone is 1 VRMS and gain is 0)
        and sens (sensitivity of output in dB(V/Pa)). These values are reported
        separately for each input. Although the dB SPL, normalized SPL and
        sensitivity of the output as measured by each input should agree, there
        will be some equipment error. So, either average them together or
        choose the most trustworthy input.
    '''
    kwargs.update(dict(gains=gain, vrms=vrms))
    result = tone_spl(engine, frequencies, *args, **kwargs)
    result['norm_spl'] = result['spl'] - gain - db(vrms)
    result['sens'] = -result['norm_spl'] - db(20e-6)
    return result


def tone_calibration(engine, frequencies, ai_channel_names, **kwargs):
    '''
    Single output calibration at a fixed frequency
    Returns
    -------
    sens : dB (V/Pa)
        Sensitivity of output in dB (V/Pa).
    '''
    kwargs.update({'engine': engine, 'frequencies': frequencies,
                   'ai_channel_names': ai_channel_names})
    output_sens = tone_sens(**kwargs)
    calibrations = {}
    for ai_channel in ai_channel_names:
        data = output_sens.loc[ai_channel]
        calibrations[ai_channel] = PointCalibration(data.index, data['sens'])
    return calibrations
