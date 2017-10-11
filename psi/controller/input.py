import logging
log = logging.getLogger(__name__)

from functools import partial
from copy import copy

import numpy as np
from scipy import signal

from atom.api import Unicode, Float, Typed, Int, Property, Enum, Bool
from enaml.core.api import d_
from enaml.workbench.api import Extension

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

    source_name = d_(Unicode())
    save = d_(Bool(False)).tag(metadata=True)

    source = Property()
    channel = Property()
    engine = Property()

    fs = Property().tag(metadata=True)
    dtype = Property().tag(metadata=True)

    def _get_source(self):
        if isinstance(self.parent, Extension):
            return None
        return self.parent

    def _set_source(self, source):
        self.set_parent(source)
        self.source_name = source.name

    def _get_fs(self):
        return self.parent.fs

    def _get_dtype(self):
        return self.parent.dtype

    def _get_channel(self):
        parent = self.parent
        while True:
            if isinstance(parent, Channel):
                return parent
            else:
                parent = parent.parent

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
        targets = [c.configure_callback(plugin) for c in self.children]
        targets.append(self.get_plugin_callback(plugin))

        # If we have only one target, no need to add another function layer
        if len(targets) == 1:
            return targets[0]
        return broadcast(*targets).send

    def get_plugin_callback(self, plugin):
        action = self.name + '_acquired'
        return lambda data: plugin.invoke_actions(action, data=data)


class ContinuousInput(Input):
    pass


class EventInput(Input):
    pass


class EpochInput(Input):

    epoch_size = Property()

    def _get_epoch_size(self):
        return self.parent.epoch_size


################################################################################
# Continuous input types
################################################################################
@coroutine
def calibrate(calibration, target):
    # TODO: hack alert here
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
        n = int(self.duration*self.parent.fs)
        return self.parent.fs/n

    def configure_callback(self, plugin):
        n = int(self.duration*self.parent.fs)
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

    duration = d_(Float()).tag(metadata=True)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        block_size = int(self.duration*self.fs)
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

    n = d_(Int()).tag(metadata=True)
    axis = d_(Int(-1)).tag(metadata=True)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return accumulate(self.n, self.axis, cb).send


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
        return self.parent.fs/self.q

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
        target(y[::q])


class Decimate(ContinuousInput):

    q = d_(Int()).tag(metadata=True)

    def _get_fs(self):
        return self.parent.fs/self.q

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
@coroutine
def extract_epochs(fs, queue, epoch_size, buffer_size, delay, target,
                   empty_queue_cb=None):
    buffer_samples = int(buffer_size*fs)
    epoch_samples = int(epoch_size*fs)
    delay_samples = int(delay*fs)

    data = (yield)
    buffer_shape = list(data.shape)
    buffer_shape[-1] = buffer_samples
    ring_buffer = np.empty(buffer_shape, dtype=data.dtype)
    next_offset = None

    # This is the timestamp of the sample at the end of the buffer. To
    # calculate the timestamp of samples at the beginning of the buffer,
    # subtract buffer_samples from t_end,
    t_end = -delay_samples

    while True:
        # Newest data is always stored at the end of the ring buffer. To make
        # room, we discard samples from the beginning of the buffer.
        samples = data.shape[-1]
        ring_buffer[..., :-samples] = ring_buffer[..., samples:]
        ring_buffer[..., -samples:] = data
        t_end += samples

        # Loop until all epochs have been extracted from buffered data.
        epochs = []
        while True:
            if (next_offset is None) and len(queue) > 0:
                offset, signal_size, key, metadata = queue.popleft()
                next_offset = int(offset*fs)
                log.trace('Next offset %d, current t_end %d', next_offset, t_end)

            if next_offset is None:
                break
            elif next_offset < (t_end - buffer_samples):
                raise SystemError('Epoch lost')
            elif (next_offset+epoch_samples) > (t_end):
                break
            else:
                # Add buffer_samples to ensure that i is indexed from the
                # beginning of the array. This ensures that we do not run into
                # the edge-case where we are using a negative indexing and we
                # want to index from, say, -10 to -0. This will result in odd
                # behavior.
                i = next_offset-t_end+buffer_samples
                epoch = {
                    'signal': ring_buffer[..., i:i+epoch_samples].copy(),
                    'key': key,
                    'metadata': metadata,
                    'offset': offset,
                }
                epochs.append(epoch)
                next_offset = None

        if len(epochs) != 0:
            target(epochs)

        if (next_offset is None) and (len(queue) == 0):
            if empty_queue_cb is not None:
                empty_queue_cb()

        data = (yield)
        log.debug('received %r samples', data.shape)


class ExtractEpochs(EpochInput):

    queue = d_(Typed(AbstractSignalQueue))
    buffer_size = d_(Float(30)).tag(metadata=True)
    epoch_size = d_(Float(8.5e-3)).tag(metadata=True)

    # This can be set to account for things such as the AO and AI filter delays
    # on the 4461. For AO at 100 kHz, the output delay is ~0.48 msec. For the
    # AI at 25e-3, the input delay is 63 samples (divide this by the
    # acquisition rate).
    delay = d_(Float(0)).tag(metadata=True)

    def configure_callback(self, plugin):
        action = self.name + '_queue_empty'
        empty_queue_cb = lambda: plugin.invoke_actions(action)
        cb = super().configure_callback(plugin)
        cb_queue = self.queue.create_connection()
        return extract_epochs(self.fs, cb_queue, self.epoch_size,
                              self.buffer_size, self.delay, cb,
                              empty_queue_cb).send



@coroutine
def reject_epochs(reject_threshold, valid_target):
    while True:
        epochs = (yield)
        valid = []
        invalid = []
        for epoch in epochs:
            # This is not an optimal approach. Normally I like to process all
            # epochs then send a bulk update. However, this ensures that we
            # preserve the correct ordering (in case that's important).
            if np.max(np.abs(epoch['signal'])) < reject_threshold:
                valid.append(epoch)
        valid_target(valid)


class RejectEpochs(EpochInput):

    threshold = d_(Float()).tag(metadata=True)

    def configure_callback(self, plugin):
        valid_cb = super().configure_callback(plugin)
        return reject_epochs(self.threshold, valid_cb).send
