import logging
log = logging.getLogger(__name__)

from functools import partial

import numpy as np
from scipy import signal

from atom.api import Unicode, Float, Typed, Int, Property, Enum
from enaml.core.api import Declarative, d_

from psi import SimpleState
from .channel import Channel


def coroutine(func):
    '''Decorator to auto-start a coroutine.'''
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        cr.next()
        return cr
    return start


@coroutine
def broadcast(targets):
    while True:
        data = (yield)
        for target in targets:
            target(data)


@coroutine
def iirfilter(N, Wn, rp, rs, btype, ftype, target):
    b, a = signal.iirfilter(N, Wn, rp, rs, btype, ftype=ftype)
    if np.any(np.abs(np.roots(a)) > 1):
        raise ValueError, 'Unstable filter coefficients'
    zf = signal.lfilter_zi(b, a)
    while True:
        y = (yield)
        y, zf = signal.lfilter(b, a, y, zi=zf)
        target(y)


@coroutine
def decimate(q, target):
    b, a = signal.cheby1(4, 0.05, 0.8/q)
    if np.any(np.abs(np.roots(a)) > 1):
        raise ValueError, 'Unstable filter coefficients'
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


@coroutine
def threshold(threshold, target):
    while True:
        samples = (yield)
        target(samples >= threshold)


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
        prior_samples = samples[..., -min_samples:]


class Input(SimpleState, Declarative):

    channel = Property().tag(transient=True)
    engine = Property().tag(transient=True)
    fs = Property()
    mode = Enum('continuous', 'event')

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
        log.debug('Configuring callback for {}'.format(self.name))
        targets = [c.configure_callback(plugin) for c in self.children]
        if self.name:
            targets.append(self.get_plugin_callback(plugin))
        return broadcast(targets).send

    def get_plugin_callback(self, plugin):
        return partial(plugin.ai_callback, self.name)


class IIRFilter(Input):

    N = d_(Int())
    #rp = d_(Float())
    #rs = d_(Float())
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
        cb = super(IIRFilter, self).configure_callback(plugin)
        return iirfilter(self.N, self.wn, None, None, self.btype,
                         self.ftype, cb).send


class Downsample(Input):

    q = d_(Int())

    def _get_fs(self):
        return self.parent.fs/self.q

    def configure_callback(self, plugin):
        cb = super(Downsample, self).configure_callback(plugin)
        return downsample(self.q, cb).send


class Decimate(Input):

    q = d_(Int())

    def _get_fs(self):
        return self.parent.fs/self.q

    def configure_callback(self, plugin):
        cb = super(Decimate, self).configure_callback(plugin)
        return decimate(self.q, cb).send


class Threshold(Input):

    threshold = d_(Float(0))

    def configure_callback(self, plugin):
        cb = super(Threshold, self).configure_callback(plugin)
        return threshold(self.threshold, cb).send


class Edges(Input):

    initial_state = d_(Int(0))
    debounce = d_(Int())
    mode = 'event'

    def get_plugin_callback(self, plugin):
        p = partial(plugin.et_callback, self.name)
        return lambda data: p(data[0], data[1]/self.fs)
    
    def configure_callback(self, plugin):
        cb = super(Edges, self).configure_callback(plugin)
        return edges(self.initial_state, self.debounce, cb).send
