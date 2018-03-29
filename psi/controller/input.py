import logging
log = logging.getLogger(__name__)

from functools import partial
from copy import copy

import numpy as np
from scipy import signal

from atom.api import (Unicode, Float, Typed, Int, Property, Enum, Bool,
                      Callable, List)
from enaml.core.api import Declarative, d_
from ..util import coroutine
from .channel import Channel
from .calibration.util import db, dbi, patodb
from .device import Device
from .queue import AbstractSignalQueue

from psi.core.enaml.api import PSIContribution


@coroutine
def broadcast(*targets):
    while True:
        data = (yield)
        for target in targets:
            target(data)


class Input(PSIContribution):

    name = d_(Unicode()).tag(metadata=True)
    label = d_(Unicode()).tag(metadata=True)

    source_name = d_(Unicode())
    source = Typed(Declarative).tag(metadata=True)
    channel = Property()
    engine = Property()

    fs = Property().tag(metadata=True)
    dtype = Property().tag(metadata=True)
    unit = Property().tag(metadata=True)
    save = d_(Bool(False)).tag(metadata=True)

    inputs = List()

    def _default_name(self):
        if self.source is not None:
            base_name = self.source.name
        else:
            base_name = self.parent.name
        return base_name + '_' + self.__class__.__name__.lower()

    def _default_label(self):
        return self.name.replace('_', ' ')

    def add_input(self, i):
        if i in self.inputs:
            return
        self.inputs.append(i)
        i.source = self

    def remove_input(self, i):
        if i not in self.inputs:
            return
        self.inputs.remove(i)
        i.source = None

    def _get_fs(self):
        return self.source.fs

    def _get_dtype(self):
        return self.source.dtype

    def _get_unit(self):
        return self.source.unit

    def _get_channel(self):
        source = self.source
        while True:
            if isinstance(source, Channel):
                return source
            else:
                source = source.source

    def _get_engine(self):
        return self.channel.engine

    def configure(self, plugin):
        cb = self.configure_callback(plugin)
        self.engine.register_ai_callback(cb, self.channel.name)

    def configure_callback(self, plugin):
        '''
        Configure callback for named inputs to ensure that they are saved to
        the data store.

        Subclasses should be sure to invoke this via super().
        '''
        log.debug('Configuring callback for {}'.format(self.name))
        targets = [c.configure_callback(plugin) for c in self.inputs]

        if plugin is not None:
            targets.append(self.get_plugin_callback(plugin))

        # If we have only one target, no need to add another function layer
        if len(targets) == 1:
            return targets[0]
        return broadcast(*targets).send

    def get_plugin_callback(self, plugin):
        action = self.name + '_acquired'
        return lambda data: plugin.invoke_actions(action, data=data)

    def add_callback(self, cb, name=''):
        callback = Callback(function=cb, name=name)
        self.add_input(callback)


class ContinuousInput(Input):
    pass


class EventInput(Input):
    pass


class EpochInput(Input):

    epoch_size = Property().tag(metadata=True)

    def _get_epoch_size(self):
        return self.source.epoch_size


class Callback(Input):

    function = Callable()

    def configure_callback(self, plugin):
        log.debug('Configuring callback for {}'.format(self.name))
        return self.function


################################################################################
# Continuous input types
################################################################################
@coroutine
def calibrate(calibration, target):
    sens = dbi(calibration.get_sens(1000))
    while True:
        data = (yield)
        target(data/sens)


class CalibratedInput(ContinuousInput):

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return calibrate(self.channel.calibration, cb).send


@coroutine
def rms(n, target):
    data = None
    while True:
        if data is None:
            data = (yield)
        else:
            data = np.concatenate((data, (yield)), axis=-1)
        while data.shape[-1] >= n:
            result = np.mean(data[..., :n]**2, axis=0)**0.5
            target(result[np.newaxis])
            data = data[..., n:]


class RMS(ContinuousInput):

    duration = d_(Float()).tag(metadata=True)

    def _get_fs(self):
        n = round(self.duration*self.source.fs)
        return self.source.fs/n

    def configure_callback(self, plugin):
        n = round(self.duration*self.source.fs)
        cb = super().configure_callback(plugin)
        return rms(n, cb).send


@coroutine
def spl(target):
    while True:
        data = (yield)
        target(patodb(data))


class SPL(ContinuousInput):

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return spl(cb).send


@coroutine
def iirfilter(N, Wn, rp, rs, btype, ftype, target):
    b, a = signal.iirfilter(N, Wn, rp, rs, btype, ftype=ftype)
    if np.any(np.abs(np.roots(a)) > 1):
        raise ValueError('Unstable filter coefficients')
    zf = signal.lfilter_zi(b, a)
    while True:
        y = (yield)
        y, zf = signal.lfilter(b, a, y, zi=zf)
        target(y)


class IIRFilter(ContinuousInput):

    N = d_(Int()).tag(metadata=True)
    btype = d_(Enum('bandpass', 'lowpass', 'highpass', 'bandstop')).tag(metadata=True)
    ftype = d_(Enum('butter', 'cheby1', 'cheby2', 'ellip', 'bessel')).tag(metadata=True)
    f_highpass = d_(Float()).tag(metadata=True)
    f_lowpass = d_(Float()).tag(metadata=True)
    wn = Property()

    def _get_wn(self):
        if self.btype == 'lowpass':
            return self.f_lowpass/(0.5*self.fs)
        elif self.btype == 'highpass':
            return self.f_highpass/(0.5*self.fs)
        else:
            return (self.f_highpass/(0.5*self.fs),
                    self.f_lowpass/(0.5*self.fs))

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return iirfilter(self.N, self.wn, None, None, self.btype,
                         self.ftype, cb).send


@coroutine
def blocked(block_size, target):
    data = []
    n = 0
    while True:
        d = (yield)
        n += d.shape[-1]
        data.append(d)
        if n >= block_size:
            merged = np.concatenate(data, axis=-1)
            while merged.shape[-1] >= block_size:
                target(merged[..., :block_size])
                merged = merged[..., block_size:]
            data = [merged]
            n = merged.shape[-1]


class Blocked(ContinuousInput):
    '''
    Chunk data based on time
    '''
    duration = d_(Float()).tag(metadata=True)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        block_size = round(self.duration*self.fs)
        return blocked(block_size, cb).send


@coroutine
def accumulate(n, axis, target):
    data = []
    while True:
        d = (yield)
        data.append(d)
        if len(data) == n:
            data = np.concatenate(data, axis=axis)
            target(data)
            data = []


class Accumulate(ContinuousInput):
    '''
    Chunk data based on number of calls
    '''
    n = d_(Int()).tag(metadata=True)
    axis = d_(Int(-1)).tag(metadata=True)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return accumulate(self.n, self.axis, cb).send


@coroutine
def capture_epoch(epoch_t0, epoch_samples, info, callback):
    '''
    Coroutine to facilitate epoch acquisition
    '''
    # This coroutine will continue until it acquires all the samples it needs.
    # It then provides the samples to the callback function and exits the while
    # loop.
    accumulated_data = []

    while True:
        tlb, data = (yield)
        samples = data.shape[-1]

        if epoch_t0 < tlb:
            # We have missed the start of the epoch. Notify the callback of this
            m = 'Missed samples for epoch of %d samples starting at %d'
            log.warn(m, start, epoch_samples)
            callback({'signal': None, 'info': info})
            break

        elif epoch_t0 <= (tlb + samples):
            # The start of the epoch is somewhere inside `data`. Find the start
            # `i` and determine how many samples `d` to extract from `data`.
            # It's possible that data does not contain the entire epoch. In
            # that case, we just pull out what we can and save it in
            # `accumulated_data`. We then update start to point to the last
            # acquired sample `i+d` and update duration to be the number of
            # samples we still need to capture.
            i = int(epoch_t0-tlb)
            d = int(min(epoch_samples, samples-i))
            accumulated_data.append(data[..., i:i+d])
            epoch_t0 += d
            epoch_samples -= d

            # Check to see if we've finished acquiring the entire epoch. If so,
            # send it to the callback.
            if epoch_samples == 0:
                accumulated_data = np.concatenate(accumulated_data, axis=-1)
                callback({'signal': accumulated_data, 'info': info})
                break


@coroutine
def extract_epochs(fs, queue, epoch_size, buffer_size, epoch_name, target,
                   empty_queue_cb=None):

    # The variable `tlb` tracks the number of samples that have been acquired
    # and reflects the lower bound of `data`. For example, if we have acquired
    # 300,000 samples, then the next chunk of data received from (yield) will
    # start at sample 300,000 (remember that Python is zero-based indexing, so
    # the first sample has an index of 0).
    tlb = 0
    epoch_coroutines = []
    prior_samples = []

    # How much historical data to keep (for retroactively capturing epochs)
    buffer_samples = int(buffer_size*fs)

    trial_index = 0

    # Since we may capture very short, rapidly occuring epochs (at, say, 80 per
    # second), I find it best to accumulate as many epochs as possible before
    # calling the next target. This list will maintain the accumulated set.
    epochs = []

    while True:
        # Wait for new data to become available
        data = (yield)

        # Check to see if more epochs have been requested. Information will be
        # provided in seconds, but we need to convert this to number of
        # samples.
        while trial_index < len(queue.uploaded):
            info = queue.uploaded[trial_index]
            trial_index += 1

            # Figure out how many samples to capture for that epoch
            t0 = round(info['t0'] * fs)
            if epoch_size:
                epoch_samples = round(epoch_size * fs)
            else:
                epoch_samples = round(info['duration'] * fs)

            epoch_coroutine = capture_epoch(t0, epoch_samples, info,
                                            epochs.append)

            try:
                # Go through the data we've been caching to facilitate
                # historical acquisition of data. If this completes without a
                # StopIteration, then we have not finished capturing the full
                # epoch.
                for prior_sample in prior_samples:
                    epoch_coroutine.send(prior_sample)
                epoch_coroutines.append(epoch_coroutine)
            except StopIteration:
                pass

        # Send the data to each coroutine. If a StopIteration occurs, this means
        # that the epoch has successfully been acquired and has been sent to the
        # callback and we can remove it. Need to operate on a copy of list since
        # it's bad form to modify a list in-place.
        for epoch_coroutine in epoch_coroutines[:]:
            try:
                epoch_coroutine.send((tlb, data))
            except StopIteration:
                epoch_coroutines.remove(epoch_coroutine)

        prior_samples.append((tlb, data))
        tlb = tlb + data.shape[-1]

        # Once the new segment of data has been processed, pass all complete
        # epochs along to the next target.
        if len(epochs) != 0:
            target(epochs[:])
            epochs[:] = []

        # Check to see if any of the cached samples are older than the specified
        # `buffer_samples` and discard them.
        while True:
            oldest_samples = prior_samples[0]
            tub = oldest_samples[0] + oldest_samples[1].shape[-1]
            if tub < (tlb-buffer_samples):
                prior_samples.pop(0)
            else:
                break

        # Check to see if any more epochs are pending
        if queue.is_empty() and \
                (len(epoch_coroutines) == 0) and \
                (trial_index == len(queue.uploaded)) and \
                empty_queue_cb is not None:
            empty_queue_cb()
            empty_queue_cb = None


#def capture(queue, target):
#    t0 = 0
#    next_offset = None
#    active = True
#
#    while True:
#        data = (yield)
#        t0 += data.shape[-1]
#
#        if next_offset is None:
#            next_offset = queue.popleft()
#
#        if next_offset is not None:
#            if t0 >= next_offset:
#                i = t0-next_offset
#
#
#
#        if t0 >= next_offset



class Capture(ContinuousInput):
    pass


@coroutine
def downsample(q, target):
    y_remainder = np.array([])
    while True:
        y = np.r_[y_remainder, (yield)]
        remainder = len(y) % q
        if remainder != 0:
            y, y_remainder = y[:-remainder], y[-remainder:]
        else:
            y_remainder = np.array([])
        target(y[::q])


class Downsample(ContinuousInput):

    q = d_(Int()).tag(metadata=True)

    def _get_fs(self):
        return self.source.fs/self.q

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return downsample(self.q, cb).send


@coroutine
def decimate(q, target):
    b, a = signal.cheby1(4, 0.05, 0.8/q)
    if np.any(np.abs(np.roots(a)) > 1):
        raise ValueError('Unstable filter coefficients')
    zf = signal.lfilter_zi(b, a)
    y_remainder = np.array([])
    while True:
        y = np.r_[y_remainder, (yield)]
        remainder = len(y) % q
        if remainder != 0:
            y, y_remainder = y[:-remainder], y[-remainder:]
        else:
            y_remainder = np.array([])
        y, zf = signal.lfilter(b, a, y, zi=zf)
        result = y[::q]
        if len(result):
            target(result)


class Decimate(ContinuousInput):

    q = d_(Int()).tag(metadata=True)

    def _get_fs(self):
        return self.source.fs/self.q

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return decimate(self.q, cb).send


@coroutine
def threshold(threshold, target):
    while True:
        samples = (yield)
        target(samples >= threshold)


class Threshold(ContinuousInput):

    threshold = d_(Float(0)).tag(metadata=True)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return threshold(self.threshold, cb).send


@coroutine
def edges(initial_state, min_samples, fs, target):
    if min_samples < 1:
        raise ValueError('min_samples must be greater than 1')
    prior_samples = np.tile(initial_state, min_samples)
    t_prior = -min_samples
    while True:
        # Wait for new data to become available
        new_samples = (yield)
        samples = np.r_[prior_samples, new_samples]
        ts_change = np.flatnonzero(np.diff(samples, axis=-1)) + 1
        ts_change = np.r_[ts_change, samples.shape[-1]]

        events = []
        for tlb, tub in zip(ts_change[:-1], ts_change[1:]):
            if (tub-tlb) >= min_samples:
                if initial_state == samples[tlb]:
                    continue
                edge = 'rising' if samples[tlb] == 1 else 'falling'
                initial_state = samples[tlb]
                ts = t_prior + tlb
                events.append((edge, ts/fs))
        events.append(('processed', t_prior/fs))
        target(events)
        t_prior += new_samples.shape[-1]
        prior_samples = samples[..., -min_samples:]


@coroutine
def average(n, target):
    data = (yield)
    axis = 0
    while True:
        while data.shape[axis] >= n:
            s = [Ellipsis]*data.ndim
            s[axis] = np.s_[:block_size]
            target(data[s].mean(axis=axis))
            s[axis] = np.s_[block_size:]
            data = data[s]
        new_data = (yield)
        data = np.concatenate((data, new_data), axis=axis)


class Average(ContinuousInput):

    n = d_(Float()).tag(metadata=True)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return average(self.n, cb).send


@coroutine
def delay(n, target):
    data = np.full(n, np.nan)
    while True:
        target(data)
        data = (yield)


class Delay(ContinuousInput):

    # This can be set to account for things such as the AO and AI filter delays
    # on the 4461. For AO at 100 kHz, the output delay is ~0.48 msec. For the
    # AI at 25e-3, the input delay is 63 samples (divide this by the
    # acquisition rate).
    delay = d_(Float(0)).tag(metadata=True)


    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        n = int(self.delay * self.fs)
        return delay(n, cb).send


################################################################################
# Event input types
################################################################################
class Edges(EventInput):

    initial_state = d_(Int(0)).tag(metadata=True)
    debounce = d_(Int()).tag(metadata=True)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return edges(self.initial_state, self.debounce, self.fs, cb).send


################################################################################
# Epoch input types
################################################################################
class ExtractEpochs(EpochInput):

    queue = d_(Typed(AbstractSignalQueue))
    buffer_size = d_(Float(0)).tag(metadata=True)
    epoch_size = d_(Float(0)).tag(metadata=True)

    complete = Bool(False)

    def mark_complete(self):
        self.complete = True

    def configure_callback(self, plugin=None):
        # If the epoch size is not set, set it to the maximum token duration
        # found in the queue. Note that this will fail if
        if self.epoch_size <= 0:
            self.epoch_size = self.queue.get_max_duration()
        if not np.isfinite(self.epoch_size):
                raise SystemError('Cannot have an infinite epoch size')

        if plugin is not None:
            # TODO: This is a hack to mark the output as active. All of this
            # shoudl be moved to the input_manifest eventually.
            plugin.invoke_actions(self.name + '_queue_start')
            def empty_queue_cb(plugin=plugin, input=self):
                plugin.invoke_actions(input.name + '_queue_end')
                input.mark_complete()
        else:
            empty_queue_cb = self.mark_complete

        cb = super().configure_callback(plugin)
        return extract_epochs(self.fs, self.queue, self.epoch_size,
                              self.buffer_size, self.name, cb,
                              empty_queue_cb).send


@coroutine
def reject_epochs(reject_threshold, status, valid_target):
    while True:
        epochs = (yield)
        valid = []

        # Find valid epochs
        for epoch in epochs:
            s = epoch['signal']
            if np.max(np.abs(epoch['signal'])) < reject_threshold:
                valid.append(epoch)

        # Send valid epochs if there are some
        if len(valid):
            valid_target(valid)

        # Update the status
        status.total += len(epochs)
        status.rejects += len(epochs)-len(valid)
        status.reject_ratio = status.rejects / status.total


class RejectEpochs(EpochInput):

    threshold = d_(Float()).tag(metadata=True)
    reject_name = d_(Unicode()).tag(metadata=True)

    total = Int()
    rejects = Int()
    reject_ratio = Float()

    def configure_callback(self, plugin):
        valid_cb = super().configure_callback(plugin)
        return reject_epochs(self.threshold, self, valid_cb).send
