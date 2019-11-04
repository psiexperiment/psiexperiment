from functools import partial
import time

import numpy as np


def acquire(engine, waveform, ao_channel_name, ai_channel_names, gain=0,
            vrms=1, repetitions=2, min_snr=None, max_thd=None, thd_harmonics=3,
            trim=0.01, iti=0.01, debug=False):
    '''
    Given a single output, measure response in multiple input channels.

    Parameters
    ----------
    TODO

    Returns
    -------
    result : array
        TODO
    '''
    if not isinstance(ao_channel_name, str):
        raise ValueError('Can only specify one output channel')

    from psi.controller.api import ExtractEpochs, FIFOSignalQueue
    from psi.controller.calibration.api import FlatCalibration

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
    ao_channel.start_trigger = ''
    for channel in ai_channels:
        channel.start_trigger = f'/{ao_channel.device_name}/ao/StartTrigger'

    samples = waveform.shape[-1]
    duration = samples / ao_fs

    # Build the signal queue
    queue = FIFOSignalQueue()
    queue.set_fs(ao_fs)
    queue.append(waveform, repetitions, iti)

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

    result = {}
    for ai_channel in ai_channels:
        # Process data from channel
        epochs = [epoch['signal'][np.newaxis] for epoch in data[ai_channel.name]]
        signal = np.concatenate(epochs)
        signal.shape = [-1, repetitions] + list(signal.shape[1:])

        if trim != 0:
            trim_samples = round(ai_channel.fs * trim)
            signal = signal[..., trim_samples:-trim_samples]

        result[ai_channel.name] = signal
        #df = pd.DataFrame(channel_result)
        #df['channel_name'] = ai_channel.name
        #result.append(df)

    return result
    #return pd.concat(result).set_index(['channel_name', 'frequency'])
