import logging
log = logging.getLogger(__name__)

from functools import partial
from collections import deque

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

    manifest = 'psi.controller.input_manifest.InputManifest'

    source_name = d_(Unicode())
    save = d_(Bool(False))

    source = Property()
    channel = Property()
    engine = Property()

    fs = Property().tag(metadata=True)
    mode = Enum('continuous', 'events', 'epochs')

    def _get_source(self):
        if isinstance(self.parent, Extension):
            return None
        return self.parent

    def _set_source(self, source):
        self.set_parent(source)
        self.source_name = source.name

    def _get_fs(self):
        return self.parent.fs

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
        if self.name:
            targets.append(self.get_plugin_callback(plugin))
        return broadcast(*targets).send

    def get_plugin_callback(self, plugin):
        action = self.name + '_acquired'
        return lambda data: plugin.invoke_actions(action, data=data)


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
                

class Blocked(Input):

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


class Accumulate(Input):

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


class Downsample(Input):

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


class Decimate(Input):

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


class Threshold(Input):

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


class Edges(Input):

    initial_state = d_(Int(0)).tag(metadata=True)
    debounce = d_(Int()).tag(metadata=True)
    mode = 'events'

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return edges(self.initial_state, self.debounce, self.fs, cb).send


@coroutine
def reject(threshold, target):
    while True:
        data = (yield)
        if np.all(np.max(np.abs(d), axis=-1) < reject_threshold):
            # TODO: what about fails?
            target(data)


class Reject(Input):

    threshold = d_(Float()).tag(metadata=True)

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

    n = d_(Float()).tag(metadata=True)

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return average(self.n, cb).send


@coroutine
def iti(fs, target):
    last_ts = 0
    while True:
        for edge, ts in (yield):
            if edge == 'rising':
                print((ts-last_ts)/fs)
                last_ts = ts


class ITI(Input):

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        return iti(self.fs, cb).send


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

        # Loop until all epochs have been extracted from buffered data.
        epochs = []
        while True:
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
                epoch = {
                    'epoch': ring_buffer[..., i:i+epoch_size].copy(),
                    'key': key,
                    'metadata': metadata
                }
                epochs.append(epoch)
                next_offset = None

        if len(epochs) != 0:
            target(epochs)
        data = (yield)
        log.debug('received %r samples', data.shape)


class ExtractEpochs(Input):

    queue = d_(Typed(AbstractSignalQueue))

    buffer_size = d_(Float(30)).tag(metadata=True)
    epoch_size = d_(Float(8.5e-3)).tag(metadata=True)

    mode = 'epochs'

    def configure_callback(self, plugin):
        cb = super().configure_callback(plugin)
        buffer_samples = int(self.buffer_size*self.fs)
        epoch_samples = int(self.epoch_size*self.fs)
        cb_queue = deque()
        self.queue.connect(cb_queue.append)
        return extract_epochs(epoch_samples, cb_queue, buffer_samples, cb).send
