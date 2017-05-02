import logging
log = logging.getLogger(__name__)

from functools import partial
from collections import deque

import numpy as np
from scipy import signal

from atom.api import Unicode, Float, Typed, Int, Property, Enum
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Extension

from psi import SimpleState
from .channel import Channel
from .calibration.util import db, dbi, patodb
from .device import Device
from .queue import AbstractSignalQueue


def coroutine(func):
    '''Decorator to auto-start a coroutine.'''
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start


@coroutine
def broadcast(targets):
    while True:
        data = (yield)
        for target in targets:
            target(data)


class Input(Device):

    source_name = d_(Unicode())

    source = Property().tag(transient=True)
    channel = Property().tag(transient=True)
    engine = Property().tag(transient=True)
    fs = Property()
    mode = Enum('continuous', 'event')

    def _get_source(self):
        if isinstance(self.parent, Extension):
            return None
        return self.parent

    def _set_source(self, source):
        self.set_parent(source)

    def _observe_parent(self, event):
        self.source_name = event['value'].name

    def _get_fs(self):
        return self.parent.fs

    def _get_channel(self):
        parent = self.parent
        while True:
            if isinstance(parent, Channel):
                return parent
            else:
                parent = parent.parent

    def _set_source(self, source):
        self.set_parent(source)

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
        if self.name:
            targets.append(self.get_plugin_callback(plugin))
        return broadcast(targets).send

    def get_plugin_callback(self, plugin):
        return partial(plugin.ai_callback, self.name)


@coroutine
def calibrate(calibration, target):
    # TODO: hack alert here
    sens = dbi(calibration.get_sens(1000))
    while True:
        data = (yield)
        target(data/sens)


class CalibratedInput(Input):

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


class RMS(Input):

    duration = d_(Float())

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


class SPL(Input):

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


class IIRFilter(Input):

    N = d_(Int())
    btype = d_(Enum('bandpass', 'lowpass', 'highpass', 'bandstop'))
    ftype = d_(Enum('butter', 'cheby1', 'cheby2', 'ellip', 'bessel'))
    f_highpass = d_(Float())
    f_lowpass = d_(Float())
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
def accumulate_segments(n, axis, target):
    data = []
    while True:
        d = (yield)
        data.append(d)
        if len(data) == n:
            data = np.concatenate(data, axis=axis)
            target(data)
            data = []


class AccumulateSegments(Input):

    n = d_(Int())
    axis = d_(Int(-1))

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return accumulate_segments(self.n, self.axis, cb).send


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


class Downsample(Input):

    q = d_(Int())

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


class Decimate(Input):

    q = d_(Int())

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


class Threshold(Input):

    threshold = d_(Float(0))

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return threshold(self.threshold, cb).send


@coroutine
def edges(initial_state, min_samples, target):
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

        for tlb, tub in zip(ts_change[:-1], ts_change[1:]):
            if (tub-tlb) >= min_samples:
                if initial_state == samples[tlb]:
                    continue
                edge = 'rising' if samples[tlb] == 1 else 'falling'
                initial_state = samples[tlb]
                target((edge, t_prior + tlb))
        t_prior += new_samples.shape[-1]
        target(('processed', t_prior))
        prior_samples = samples[..., -min_samples:]


class Edges(Input):

    initial_state = d_(Int(0))
    debounce = d_(Int())
    mode = 'event'

    def get_plugin_callback(self, plugin):
        p = partial(plugin.et_callback, self.name)
        return lambda data: p(data[0], data[1]/self.fs)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return edges(self.initial_state, self.debounce, cb).send


@coroutine
def reject(threshold, target):
    while True:
        data = (yield)
        if np.all(np.max(np.abs(d), axis=-1) < reject_threshold):
            # TODO: what about fails?
            target(data)


class Reject(Input):

    threshold = d_(Float())

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return reject(self.threshold, cb).send


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


class Average(Input):

    n = d_(Float())

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return average(self.n, cb).send


@coroutine
def notify_epoch_acquired(plugin, name, target):
    action = '{}_acquired'.format(name)
    while True:
        epoch, key, metadata = (yield)
        plugin.invoke_actions(action, key=key, metadata=metadata)


class BaseEpoch(Input):

    def configure_callback(self, plugin):
        # This tacks on a notifier that triggers an action every time an epoch
        # is acquired.
        cb = super().configure_callback(plugin)
        return notify_epoch_acquired(plugin, self.name, cb).send

    def get_plugin_callback(self, plugin):
        return partial(plugin.ai_epoch_callback, self.name)


@coroutine
def extract_epochs(epoch_size, queue, buffer_size, target):
    buffer_size = int(buffer_size)
    data = (yield)
    buffer_shape = list(data.shape)
    buffer_shape[-1] = buffer_size
    ring_buffer = np.empty(buffer_shape, dtype=data.dtype)
    next_offset = None
    t0 = -buffer_size

    while True:
        # Newest data is always stored at the end of the ring buffer. To make
        # room, we discard samples from the beginning of the buffer.
        samples = data.shape[-1]
        ring_buffer[..., :-samples] = ring_buffer[..., samples:]
        ring_buffer[..., -samples:] = data
        t0 += samples

        while True:
            log.debug('looping')
            if (next_offset is None) and len(queue) > 0:
                next_offset, key, metadata = queue.popleft()
                log.debug('Next offset %d, current t0 %d', next_offset, t0)
            elif next_offset is None:
                break
            elif next_offset < t0:
                raise SystemError('Epoch lost')
            elif (next_offset+epoch_size) > (t0+buffer_size):
                break
            else:
                i = next_offset-t0
                epoch = ring_buffer[..., i:i+epoch_size].copy()
                target((epoch, key, metadata))
                next_offset = None

        data = (yield)
        log.debug('received %r samples', data.shape)


class ExtractEpochs(BaseEpoch):

    manifest = 'psi.controller.input_manifest.QueuedEpochInputManifest'

    queue = d_(Typed(AbstractSignalQueue))

    buffer_size = d_(Float(30))
    epoch_size = d_(Float(8.5e-3))

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        buffer_samples = int(self.buffer_size*self.fs)
        epoch_samples = int(self.epoch_size*self.fs)
        cb_queue = deque()
        self.queue.connect(cb_queue.append)
        return extract_epochs(epoch_samples, cb_queue, buffer_samples, cb).send
