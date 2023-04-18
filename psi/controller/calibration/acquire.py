import logging
log = logging.getLogger(__name__)

from functools import partial
import time

import numpy as np
import pandas as pd

from psiaudio.pipeline import concat
from psiaudio.queue import FIFOSignalQueue
from psi.controller.api import ExtractEpochs


def _reindex(x):
    # Add a "repeat" column indicating the repetition number of the stimulus.
    x['repeat'] = np.arange(len(x))
    return x


def acquire(cal_engine, ao_channel_name, ai_channel_names, setup_queue_cb,
            epoch_size, trim=0):
    '''
    Utility function to facilitate acquisition of calibration signals
    '''
    if not isinstance(ao_channel_name, str):
        raise ValueError('Can only specify one output channel')

    cal_engine = cal_engine.clone([ao_channel_name] + ai_channel_names)

    ao_channel = cal_engine.get_channel(ao_channel_name)
    ai_channels = [cal_engine.get_channel(name) for name in ai_channel_names]

    ao_fs = ao_channel.fs
    ai_fs = ai_channels[0].fs

    # Ensure that input channels are synced to the output channel
    for channel in ai_channels:
        channel.sync_start(ao_channel)

    # Set up the queued output
    queue = FIFOSignalQueue()
    queue.set_fs(ao_fs)
    output = ao_channel.add_queued_epoch_output(queue, auto_decrement=True)
    output.activate(0)

    setup_queue_cb(ao_channel, queue)
    data = {ai_channel: [] for ai_channel in ai_channels}

    def accumulate(epochs, epoch):
        epochs.extend(epoch)

    to_remove = []
    for ai_channel in ai_channels:
        epoch_input = ExtractEpochs(epoch_size=epoch_size)
        cb = partial(accumulate, data[ai_channel])
        epoch_input.add_callback(cb)

        queue.connect(epoch_input.added_queue.append, 'added')
        queue.connect(epoch_input.source_complete, 'empty')
        ai_channel.add_input(epoch_input)
        to_remove.append((ai_channel, epoch_input))

    cal_engine.configure()
    cal_engine.start()
    while True:
        if queue.is_empty() and epoch_input.complete:
            break
        time.sleep(0.1)
    time.sleep(0.1)
    cal_engine.stop()

    result = {}
    for ai_channel, epochs in data.items():
        epochs = concat(epochs, axis='epoch')
        keys = pd.DataFrame(epochs.metadata)
        grouping = keys.columns.tolist()
        grouping.remove('t0')
        keys = keys.groupby(grouping, group_keys=False).apply(_reindex)
        keys.index.name = 'epoch'
        if trim != 0:
            trim_samples = round(ai_channel.fs * trim)
            epochs = epochs[..., trim_samples:-trim_samples]

        col_index = pd.MultiIndex.from_frame(keys.reset_index())
        t = np.arange(epochs.shape[-1]) / ai_channel.fs
        time_index = pd.Index(t, name='time')
        result[ai_channel] = pd.DataFrame(epochs[:, 0], index=col_index, columns=time_index)

    return result
