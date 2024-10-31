import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from .acquire import acquire
from psiaudio import util
from psiaudio.calibration import FlatCalibration, PointCalibration
from psiaudio.stim import ClickFactory, SilenceFactory


def click_power(engines, ao_channel_name, ai_channel_names, gain=0, vrms=1,
                discard=2, repetitions=10, min_snr=None, duration=100e-6,
                iti=0.01, trim=0, debug=False):
    '''
    Given a single output, measure response in multiple input channels.

    Parameters
    ----------

    Returns
    -------
    result : pandas DataFrame
        Dataframe will be indexed by output channel name. Columns
        will be rms (in V), snr (in DB) and thd (in percent).
    '''
    calibration = FlatCalibration.as_attenuation(vrms=vrms)
    factory_kw = {
        'calibration': calibration,
        'polarity': 1,
        'level': gain,
        'duration': duration,
    }

    def setup_queue_cb(ao_channel, queue):
        nonlocal factory_kw
        nonlocal duration

        samples = int(ao_channel.fs * duration)

        factory = ClickFactory(ao_channel.fs, **factory_kw)
        click_waveform = factory.next(samples)
        queue.append(click_waveform, repetitions, iti, metadata={'gain': gain})

        factory = SilenceFactory()
        waveform = factory.next(samples)
        queue.append(waveform, repetitions, iti, metadata={'gain': -400})

    recording = acquire(engines, ao_channel_name, ai_channel_names,
                        setup_queue_cb, duration + iti, trim)

    result = {}
    waveforms = {}
    stimulus = {}
    for ai_channel, signal in recording.items():
        mean_signal = signal.groupby('gain').mean()
        samples = int(round(ai_channel.fs * (duration + iti)))
        factory = ClickFactory(ai_channel.fs, **factory_kw)
        click_waveform = factory.next(samples)

        stimulus_psd = util.psd_df(click_waveform, ai_channel.fs)
        mean_psd = util.psd_df(mean_signal, ai_channel.fs)

        result[ai_channel.name] = pd.DataFrame({
            'click_rms': mean_psd.loc[gain],
            'silence_rms': mean_psd.loc[-400],
            'stimulus_rms': stimulus_psd,
            'snr': util.db(mean_psd.loc[gain] / mean_psd.loc[-400]),
        })
        waveforms[ai_channel.name] = signal

        t = np.arange(len(click_waveform)) / ai_channel.fs
        stimulus[ai_channel.name] = pd.Series(click_waveform, index=pd.Index(t, name='time'))

    result = pd.concat(result, names=['channel'])
    stimulus = pd.DataFrame(stimulus).T
    stimulus.index.name = 'channel'
    result.attrs['waveforms'] = pd.concat(waveforms, names=['channel'])
    result.attrs['fs'] = {c.name: c.fs for c in recording}
    result.attrs['stimulus'] = stimulus
    return result


def click_spl(engines, *args, **kwargs):
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
    result = click_power(engines, *args, **kwargs)

    if not isinstance(engines, (tuple, list)):
        engines = [engines]

    channel_map = {}
    for engine in engines:
        for channel in engine.get_channels(active=False):
            channel_map[channel.name] = channel

    def map_spl(df, channel_map):
        channel_name = df.name
        frequency = df.index.get_level_values('frequency')
        channel = channel_map[channel_name]
        df['click_spl'] = channel.calibration.get_db(frequency, df['click_rms'])
        df['silence_spl'] = channel.calibration.get_db(frequency, df['silence_rms'])
        df.attrs = {}
        return df

    new_result = result.groupby('channel', group_keys=False) \
        .apply(map_spl, channel_map=channel_map)
    new_result.attrs.update(result.attrs)
    return new_result


def click_sens(gain=-40, vrms=1, **kwargs):
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
        SPL, that would be generated assuming the click is 1 VRMS and gain is 0)
        and sens (sensitivity of output in dB(V/Pa)). These values are reported
        separately for each input. Although the dB SPL, normalized SPL and
        sensitivity of the output as measured by each input should agree, there
        will be some equipment error. So, either average them together or
        choose the most trustworthy input.
    '''
    kwargs.update(dict(gain=gain, vrms=vrms))
    result = click_spl(**kwargs)

    # Calculate the overall SPL of the click
    spl = result.groupby('channel')['click_spl'].apply(util.rms_rfft_db)
    nf = result.groupby('channel')['silence_spl'].apply(util.rms_rfft_db)

    norm_spl = spl - (gain + util.db(vrms))

    sens = pd.DataFrame({
        'spl': spl,
        'noise_floor': nf,
        'norm_spl': norm_spl,
    })
    sens.attrs['SPL'] = result.copy()
    sens.attrs['SPL'].attrs = {}
    sens.attrs.update(result.attrs)
    return sens
