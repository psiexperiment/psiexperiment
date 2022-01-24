from functools import partial
import time

import numpy as np
import pandas as pd

from psi.controller.calibration.calibration import FlatCalibration
from psiaudio.stim import ChirpFactory, SilenceFactory

from .calibration import InterpCalibration
from . import util


def chirp_power(engine, ao_channel_name, ai_channel_names, start_frequency=500,
                end_frequency=50000, gain=0, vrms=1, repetitions=64,
                duration=20e-3, iti=0.001, debug=False):
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
    from psi.controller.api import ExtractEpochs, FIFOSignalQueue
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

    # Build the signal queue
    queue = FIFOSignalQueue()
    queue.set_fs(ao_fs)

    # Create and add the chirp
    factory = ChirpFactory(ao_fs, start_frequency, end_frequency, duration,
                           gain, calibration)
    chirp_waveform = factory.next(samples)
    queue.append(chirp_waveform, repetitions, iti, metadata={'gain': gain})

    # Create and add silence
    factory = SilenceFactory(ao_fs, calibration)
    waveform = factory.next(samples)
    queue.append(waveform, repetitions, iti, metadata={'gain': -400})

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
        epoch_input = ExtractEpochs(epoch_size=duration+iti)
        queue.connect(epoch_input.queue.append)
        epoch_input.add_callback(cb)
        ai_channel.add_input(epoch_input)
        ai_channel.add_callback(samples[ai_channel.name].append)

    cal_engine.start()
    while not epoch_input.complete:
        time.sleep(0.1)
    cal_engine.stop()

    result_waveforms = {}
    result_psd = {}
    for ai_channel in ai_channels:
        epochs = data[ai_channel.name]
        waveforms = [e['signal'] for e in epochs]
        keys = [e['info']['metadata'] for e in epochs]
        keys = pd.DataFrame(keys)
        keys.index.name = 'epoch'
        keys = keys.set_index(['gain'], append=True)
        keys.index = keys.index.swaplevel('epoch', 'gain')

        waveforms = np.vstack(waveforms)
        t = np.arange(waveforms.shape[-1]) / ai_channel.fs
        time_index = pd.Index(t, name='time')
        waveforms = pd.DataFrame(waveforms, index=keys.index,
                                 columns=time_index)
        mean_waveforms = waveforms.groupby('gain').mean()

        samples = int(round(ai_channel.fs * (duration + iti)))
        factory = ChirpFactory(ai_channel.fs, start_frequency, end_frequency,
                               duration, gain, calibration)
        chirp_waveform = factory.next(samples)

        chirp_psd = util.psd_df(chirp_waveform, ai_channel.fs)
        mean_psd = util.psd_df(mean_waveforms, ai_channel.fs)

        result_psd[ai_channel.name] = pd.DataFrame({
            'rms': mean_psd.loc[gain],
            'chirp_rms': chirp_psd,
            'snr': util.db(mean_psd.loc[gain] / mean_psd.loc[-400]),
        })
        #result_waveforms[ai_channel.name] = waveforms

    #result_waveforms = pd.concat(result_waveforms.values(),
    #                             keys=result_waveforms.keys(),
    #                             names=['channel'])

    result_psd = pd.concat(result_psd.values(), keys=result_psd.keys(),
                           names=['channel'])

    return result_psd


def chirp_spl(engine, **kwargs):

    def map_spl(series, engine):
        channel_name, = series.index.get_level_values('channel').unique()
        channel = engine.get_channel(channel_name)
        frequency = series.index.get_level_values('frequency')
        series['spl'] = channel.calibration.get_spl(frequency, series['rms'])
        return series

    result = chirp_power(engine, **kwargs)
    return result.groupby('channel').apply(map_spl, engine=engine)


def chirp_sens(engine, gain=-40, vrms=1, **kwargs):
    result = chirp_spl(engine, gain=gain, vrms=vrms, **kwargs)
    result['norm_spl'] = result['spl'] - util.db(result['chirp_rms'])
    result['sens'] = -result['norm_spl'] - util.db(20e-6)
    return result


def chirp_calibration(ai_channel_names, **kwargs):
    kwargs.update({'ai_channel_names': ai_channel_names})
    output_sens = chirp_sens(**kwargs)
    calibrations = {}
    for ai_channel in ai_channel_names:
        data = output_sens.loc[ai_channel]
        calibrations[ai_channel] = InterpCalibration(data.index, data['sens'])
    return calibrations
