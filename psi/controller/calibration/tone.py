import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from .acquire import acquire
from psiaudio.util import db, process_tone, tone_power_conv, tone_phase_conv
from psiaudio.calibration import FlatCalibration, PointCalibration
from psiaudio.stim import ToneFactory, SilenceFactory


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
    frequencies = np.asarray(frequencies)
    if np.isscalar(gains):
        gains = [gains] * len(frequencies)

    def setup_queue_cb(ao_channel, queue):
        nonlocal vrms
        nonlocal duration
        nonlocal frequencies
        nonlocal gains

        calibration = FlatCalibration.as_attenuation(vrms=vrms)
        samples = int(ao_channel.fs * duration)

        # Build the signal queue
        max_sf = 0
        for frequency, gain in zip(frequencies, gains):
            factory = ToneFactory(ao_channel.fs, level=gain,
                                  frequency=frequency, calibration=calibration)
            waveform = factory.next(samples)
            md = {'gain': gain, 'frequency': frequency}
            queue.append(waveform, repetitions, iti, metadata=md)
            sf = calibration.get_sf(frequency, gain) * np.sqrt(2)
            max_sf = max(max_sf, sf)
        ao_channel.expected_range = (-max_sf*1.1, max_sf*1.1)

        factory = SilenceFactory()
        waveform = factory.next(samples)
        md = {'gain': -400, 'frequency': 0}
        queue.append(waveform, repetitions, iti, metadata=md)

    recording = acquire(engine, ao_channel_name, ai_channel_names,
                        setup_queue_cb, duration, trim)

    result = []
    for ai_channel, signal in recording.items():
        silence = signal.query('gain == -400')
        signal = signal.query('gain != -400')

        channel_result = []
        for f, s in signal.groupby('frequency'):
            f_result = process_tone(ai_channel.fs, s.values, f, min_snr,
                                    max_thd, thd_harmonics, silence.values)
            f_result['frequency'] = f
            channel_result.append(f_result)

        df = pd.DataFrame(channel_result)
        df['channel_name'] = ai_channel.name
        df['input_channel_gain'] = ai_channel.gain
        result.append(df)

    result = pd.concat(result).set_index(['channel_name', 'frequency'])
    result.attrs['waveforms'] = signal
    result.attrs['fs'] = {c.name: c.fs for c in recording}
    return result


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
        spl = channel.calibration.get_db(frequency, series['rms'])
        series['spl'] = spl
        return series

    new_result = result.apply(map_spl, axis=1, args=(engine,))
    new_result.attrs.update(result.attrs)
    return new_result


def tone_sens(engine, frequencies, gains=-40, vrms=1, **kwargs):
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
    kwargs.update(dict(gains=gains, vrms=vrms))
    result = tone_spl(engine, frequencies, **kwargs)

    # Need to reshape for the math in case we provided a different gain for each frequency.
    spl = result['spl'].unstack('channel_name')
    norm_spl = spl.subtract(gains + db(vrms), axis=0)
    norm_spl = norm_spl.stack().reorder_levels(result.index.names)

    # psiaudio calibration units are in dB(Pa/20e-6/V), so this is basically
    # the normalized SPL (i.e. SPL produced by a 1 Vrms sine wave). How
    # convenient.
    result['sens'] = result['norm_spl'] = norm_spl

    result['gain'] = gains
    result['vrms'] = vrms
    for k, v in kwargs.items():
        if k in ('ao_channel_name', 'ai_channel_names'):
            continue
        result[k] = v

    return result
