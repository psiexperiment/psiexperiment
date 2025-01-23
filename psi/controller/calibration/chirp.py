import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from psiaudio.stim import ChirpFactory, SilenceFactory
from psiaudio.calibration import FlatCalibration, InterpCalibration
from psiaudio import util



def chirp_power(engines, ao_channel_name, ai_channel_names,
                start_frequency=500, end_frequency=50000, gain=0, vrms=1,
                repetitions=64, duration=20e-3, iti=0.001, debug=False):
    '''
    Given a single output, measure response in multiple input channels using
    chirp.

    Parameters
    ----------
    TODO

    Returns
    -------
    result : pandas DataFrame
        Dataframe will be indexed by output channel name and frequency. Columns
        will be rms (in V), snr (in DB) and thd (in percent).
    '''
    # Avoid circular import
    from .acquire import acquire

    calibration = FlatCalibration.as_attenuation(vrms=vrms)
    factory_kw = {
        'start_frequency': start_frequency,
        'end_frequency': end_frequency,
        'duration': duration,
        'level': gain,
        'calibration': calibration,
    }

    def setup_queue_cb(ao_channel, queue):
        nonlocal duration
        nonlocal gain
        nonlocal factory_kw
        nonlocal repetitions
        nonlocal iti

        samples = int(ao_channel.fs * duration)

        # Create and add the chirp
        factory = ChirpFactory(ao_channel.fs, **factory_kw)
        chirp_waveform = factory.next(samples)
        queue.append(chirp_waveform, repetitions, iti, metadata={'gain': gain})

        # Create and add silence
        factory = SilenceFactory()
        waveform = factory.next(samples)
        queue.append(waveform, repetitions, iti, metadata={'gain': -400})

    recording = acquire(engines, ao_channel_name, ai_channel_names,
                        setup_queue_cb, duration + iti, 0)

    result = {}
    waveforms = {}
    for ai_channel, signal in recording.items():
        mean_signal = signal.groupby('gain').mean()
        samples = int(round(ai_channel.fs * (duration + iti)))
        factory = ChirpFactory(ai_channel.fs, **factory_kw)
        chirp_waveform = factory.next(samples)

        chirp_psd = util.psd_df(chirp_waveform, ai_channel.fs)
        mean_psd = util.psd_df(mean_signal, ai_channel.fs)

        result[ai_channel.name] = pd.DataFrame({
            'rms': mean_psd.loc[gain],
            'chirp_rms': chirp_psd,
            'snr': util.db(mean_psd.loc[gain] / mean_psd.loc[-400]),
        })
        waveforms[ai_channel.name] = signal

    waveforms = pd.concat(waveforms, names=['channel'])
    result = pd.concat(result, names=['channel'])
    if debug:
        result.attrs['waveforms'] = waveforms
        result.attrs['fs'] = {c.name: c.fs for c in recording}
    return result


def chirp_spl(engines, **kwargs):

    channel_map = {}
    for engine in engines:
        for channel in engine.get_channels(active=False):
            channel_map[channel.name] = channel

    def map_spl(series, engine):
        channel_name, = series.index.get_level_values('channel').unique()
        channel = channel_map[channel_name]
        frequency = series.index.get_level_values('frequency')
        series['spl'] = channel.calibration.get_db(frequency, series['rms'])
        return series

    result = chirp_power(engines, **kwargs)
    new_result = result.groupby('channel', group_keys=False) \
        .apply(map_spl, engine=engine)
    new_result.attrs.update(result.attrs)
    return new_result


def chirp_sens(engines, gain=-40, vrms=1, **kwargs):
    result = chirp_spl(engines, gain=gain, vrms=vrms, **kwargs)
    result['sens'] = result['norm_spl'] = result['spl'] - util.db(result['chirp_rms'])
    return result
