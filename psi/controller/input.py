import logging

log = logging.getLogger(__name__)

from collections import deque
from functools import partial
#from queue import Empty, Queue

import numpy as np
from scipy import signal

from atom.api import (Str, Float, Typed, Int, Property, Enum, Bool,
                      Callable, List, Tuple, set_default)
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_
import xarray as xr

from psiaudio.calibration import FlatCalibration
from psiaudio.pipeline import coroutine, extract_epochs
from psiaudio.util import dbi, patodb

from .channel import Channel

from psi.core.enaml.api import PSIContribution


@coroutine
def broadcast(*targets):
    while True:
        data = (yield)
        for target in targets:
            target(data)


class Input(PSIContribution):

    name = d_(Str()).tag(metadata=True)
    label = d_(Str()).tag(metadata=True)
    force_active = d_(Bool(False)).tag(metadata=True)

    source_name = d_(Str()).tag(metadata=True)
    source = d_(Typed(Declarative).tag(metadata=True), writable=False)
    channel = Property().tag(metadata=True)
    engine = Property().tag(metadata=True)

    fs = Property().tag(metadata=True)
    dtype = Property().tag(metadata=True)
    unit = Property().tag(metadata=True)
    calibration = Property().tag(metadata=True)
    active = Property().tag(metadata=True)

    inputs = List().tag(metadata=True)

    configured = Bool(False)

    def _default_name(self):
        if self.source is not None:
            base_name = self.source.name
        else:
            base_name = self.parent.name
        return f'{base_name}_{self.__class__.__name__}'

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
        if self.source is None:
            raise AttributeError(f'No source specified for input {self.name}')
        return self.source.fs

    def _get_dtype(self):
        return self.source.dtype

    def _get_calibration(self):
        return self.source.calibration

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

    def configure(self):
        cb = self.configure_callback()
        self.engine.register_ai_callback(cb, self.channel.name)
        self.configured = True

    def configure_callback(self):
        targets = [i.configure_callback() for i in self.inputs if i.active]
        log.debug('Configured callback for %s with %d targets', self.name,
                  len(targets))
        if len(targets) == 1:
            return targets[0]
        # If we have more than one target, need to add a broadcaster
        return broadcast(*targets).send

    def add_callback(self, cb):
        if self.configured:
            m = f'{self.name} already configured. Cannot add callback.'
            raise ValueError(m)

        callback = Callback(function=cb)
        self.add_input(callback)

    def _get_active(self):
        return self.force_active or any(i.active for i in self.inputs)


class ContinuousInput(Input):
    pass


class EventInput(Input):
    pass


class EpochInput(Input):

    duration = Property().tag(metadata=True)

    def _get_duration(self):
        return self.source.duration


class Callback(Input):
    function = d_(Callable())

    def _default_name(self):
        return str(self.function)

    def configure_callback(self):
        log.debug('Configuring callback for {}'.format(self.name))
        return self.function

    def _get_active(self):
        return True


################################################################################
# Continuous input types
################################################################################
@coroutine
def custom_input(function, target):
    while True:
        data = (yield)
        function(data, target)


class CustomInput(Input):
    function = d_(Callable())

    def configure_callback(self):
        cb = super().configure_callback()
        return custom_input(self.function, cb).send


@coroutine
def calibrate(calibration, target):
    sens = dbi(calibration.get_sens(1000))
    log.debug('Setting sensitivity for CalibratedInput to %f', sens)
    while True:
        data = (yield)
        target(data * sens)


class CalibratedInput(ContinuousInput):
    '''
    Applies calibration to input

    Currently assumes that calibration is flat across frequency.

    If input is from a microphone and the microphone calibration transforms
    from Vrms to Pascals, then the output of this block will be in Pascals.
    '''
    def _get_calibration(self):
        # Input is now calibrated, and no additional transforms need to be
        # performed by downstream inputs. Note that the calibration from the
        # source for this input is used (i.e., not *this* calibration). So,
        # calibration is now 1 unit per 1 Pa. (i.e., dB(1/1Pa) gives us a
        # sensitivity of 0). This works beause you can show that:
        # >>> FlatCalibration(sensitivity=0).get_spl(1)
        # 93.9794
        # Which is consistent with 1 Pa = 94 dB SPL
        return FlatCalibration(sensitivity=0)

    def configure_callback(self):
        cb = super().configure_callback()
        log.debug('Configuring CalibratedInput %s', self.name)
        return calibrate(self.source.calibration, cb).send


@coroutine
def rms(n, target):
    data = None
    while True:
        if data is None:
            data = (yield)
        else:
            data = np.concatenate((data, (yield)), axis=-1)
        while data.shape[-1] >= n:
            result = np.mean(data[..., :n] ** 2, axis=0) ** 0.5
            target(result[np.newaxis])
            data = data[..., n:]


class RMS(ContinuousInput):
    duration = d_(Float()).tag(metadata=True)

    def _get_fs(self):
        n = round(self.duration * self.source.fs)
        return self.source.fs / n

    def configure_callback(self):
        n = round(self.duration * self.source.fs)
        cb = super().configure_callback()
        return rms(n, cb).send


@coroutine
def spl(target, sens):
    v_to_pa = dbi(sens)
    while True:
        data = (yield)
        data /= v_to_pa
        spl = patodb(data)
        target(spl)


class SPL(ContinuousInput):

    def configure_callback(self):
        cb = super().configure_callback()
        sens = self.calibration.get_sens(1000)
        return spl(cb, sens).send


@coroutine
def iirfilter(N, Wn, rp, rs, btype, ftype, target):
    b, a = signal.iirfilter(N, Wn, rp, rs, btype, ftype=ftype)
    if np.any(np.abs(np.roots(a)) > 1):
        raise ValueError('Unstable filter coefficients')

    # Initialize the state of the filter and scale it by y[0] to avoid a
    # transient.
    zi = signal.lfilter_zi(b, a)
    y = (yield)
    zo = zi * y[0]

    while True:
        #y, zo = signal.lfilter(b, a, y, zi=zo)
        y = xd.apply_ufunc(signal.lfilter, b, a, y, kwargs={'zi': zo})
        target(y)
        y = (yield)


class IIRFilter(ContinuousInput):
    # Allows user to deactivate the filter entirely during configuration if
    # desired. Ideally we could just remove it from the graph, but it seems a
    # bit tricky to do so as some other components may be looking for the
    # output of this block.
    passthrough = d_(Bool(False)).tag(metadata=True)

    N = d_(Int(1)).tag(metadata=True)
    btype = d_(Enum('bandpass', 'lowpass', 'highpass', 'bandstop')).tag(
        metadata=True)
    ftype = d_(Enum('butter', 'cheby1', 'cheby2', 'ellip', 'bessel')).tag(
        metadata=True)
    f_highpass = d_(Float()).tag(metadata=True)
    f_lowpass = d_(Float()).tag(metadata=True)
    wn = Property().tag(metadata=True)

    def _get_wn(self):
        log.debug('%s filter from %f to %f at Fs=%f', self.name,
                  self.f_highpass, self.f_lowpass, self.fs)
        if self.btype == 'lowpass':
            log.debug('Lowpass at %r (fs=%r)', self.f_lowpass, self.fs)
            return self.f_lowpass / (0.5 * self.fs)
        elif self.btype == 'highpass':
            log.debug('Highpass at %r (fs=%r)', self.f_lowpass, self.fs)
            return self.f_highpass / (0.5 * self.fs)
        else:
            log.debug('Bandpass %r to %r (fs=%r)', self.f_highpass,
                      self.f_lowpass, self.fs)
            return (self.f_highpass / (0.5 * self.fs),
                    self.f_lowpass / (0.5 * self.fs))

    def configure_callback(self):
        cb = super().configure_callback()
        if self.passthrough:
            return cb
        return iirfilter(self.N, self.wn, None, None, self.btype, self.ftype,
                         cb).send


@coroutine
def blocked(block_size, target):
    data = []
    n = 0

    while True:
        d = (yield)
        if d is Ellipsis:
            data = []
            target(d)
            continue

        n += d.shape[-1]
        data.append(d)
        if n >= block_size:
            merged = concatenate(data, axis=-1)
            while merged.shape[-1] >= block_size:
                block = merged[..., :block_size]
                block.metadata['block_size'] = block_size
                target(block)
                merged = merged[..., block_size:]
            data = [merged]
            n = merged.shape[-1]


class Blocked(ContinuousInput):
    '''
    Chunk data based on time
    '''
    duration = d_(Float()).tag(metadata=True)

    def configure_callback(self):
        if self.duration <= 0:
            m = 'Duration for {} must be > 0'.format(self.name)
            raise ValueError(m)
        cb = super().configure_callback()
        block_size = round(self.duration * self.fs)
        return blocked(block_size, cb).send


@coroutine
def accumulate(n, axis, newaxis, status_cb, target):
    data = []
    while True:
        d = (yield)
        if d is Ellipsis:
            data = []
            target(d)
            continue

        if newaxis:
            data.append(d[np.newaxis])
        else:
            data.append(d)
        if len(data) == n:
            data = concatenate(data, axis=axis)
            target(data)
            data = []

        if status_cb is not None:
            status_cb(len(data))


class Accumulate(ContinuousInput):
    '''
    Chunk data based on number of calls
    '''
    n = d_(Int()).tag(metadata=True)
    axis = d_(Int(-1)).tag(metadata=True)
    newaxis = d_(Bool(False)).tag(metadata=True)

    status_cb = d_(Callable(lambda x: None))

    def configure_callback(self):
        cb = super().configure_callback()
        return accumulate(self.n, self.axis, self.newaxis, self.status_cb,
                          cb).send


@coroutine
def capture(fs, queue, target):
    s0 = 0
    t_start = None  # Time, in seconds, of capture start
    s_next = None  # Sample number for capture

    while True:
        # Wait for new data to come in
        data = (yield)
        try:
            # We've recieved a new command. The command will either be None
            # (i.e., no more acquisition for a bit) or a floating-point value
            # (indicating when next acquisition should begin).
            info = queue.popleft()
            if info is not None:
                t_start = info['t0']
                s_next = round(t_start * fs)
                target(Ellipsis)
                log.error('Starting capture at %f', t_start)
            elif info is None:
                log.debug('Ending capture')
                s_next = None
            else:
                raise ValueError('Unsupported queue input %r', info)
        except IndexError:
            pass

        if (s_next is not None) and (s_next >= s0):
            i = s_next - s0
            if i < data.shape[-1]:
                d = data[i:]
                d.metadata['capture'] = t_start
                target(d)
                s_next += d.shape[-1]

        s0 += data.shape[-1]


class Capture(ContinuousInput):

    #: This is used internally by the CaptureManifest to notify Capture that
    #: new trials have been enqueued.
    queue = Typed(deque, ())

    #: The event that indicates beginning of capture. If left blank, the
    #: programmer is responsible for explicitly hooking up the event.
    start_event = d_(Str())

    def configure_callback(self):
        cb = super().configure_callback()
        return capture(self.fs, self.queue, cb).send


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
        result = y[::q]
        if len(result):
            target(result)


class Downsample(ContinuousInput):
    q = d_(Int()).tag(metadata=True)

    def _get_fs(self):
        return self.source.fs / self.q

    def configure_callback(self):
        cb = super().configure_callback()
        return downsample(self.q, cb).send


@coroutine
def decimate(q, target):
    b, a = signal.cheby1(4, 0.05, 0.8 / q)
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
        return self.source.fs / self.q

    def configure_callback(self):
        cb = super().configure_callback()
        return decimate(self.q, cb).send


@coroutine
def discard(discard_samples, cb):
    to_discard = discard_samples
    while True:
        samples = (yield)
        if samples is Ellipsis:
            # Restart the pipeline
            to_discard = discard_samples
            cb(samples)
            continue

        samples.metadata['discarded'] = discard_samples
        if to_discard == 0:
            cb(samples)
        elif samples.shape[-1] <= to_discard:
            to_discard -= samples.shape[-1]
        elif samples.shape[-1] > to_discard:
            samples = samples[..., to_discard:]
            to_discard = 0
            cb(samples)


class Discard(ContinuousInput):
    duration = d_(Float()).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        samples = round(self.duration * self.fs)
        return discard(samples, cb).send


@coroutine
def threshold(threshold, target):
    while True:
        samples = (yield)
        target(samples >= threshold)


class Threshold(ContinuousInput):
    threshold = d_(Float(0)).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        return threshold(self.threshold, cb).send


@coroutine
def average(n, target):
    data = (yield)
    axis = 0
    while True:
        while data.shape[axis] >= n:
            s = [Ellipsis] * data.ndim
            s[axis] = np.s_[:block_size]
            target(data[s].mean(axis=axis))
            s[axis] = np.s_[block_size:]
            data = data[s]
        new_data = (yield)
        data = np.concatenate((data, new_data), axis=axis)


class Average(ContinuousInput):
    n = d_(Float()).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
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

    def configure_callback(self):
        cb = super().configure_callback()
        n = round(self.delay * self.fs)
        return delay(n, cb).send


@coroutine
def transform(function, target):
    while True:
        data = (yield)
        transformed_data = function(data)
        target(transformed_data)


class Transform(ContinuousInput):
    function = d_(Callable())

    def configure_callback(self):
        cb = super().configure_callback()
        return transform(self.function, cb).send


class Coroutine(Input):
    coroutine = d_(Callable())
    args = d_(Tuple())
    force_active = set_default(True)

    def configure_callback(self):
        cb = super().configure_callback()
        return self.coroutine(*self.args, cb).send


################################################################################
# Event input types
################################################################################
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
            if (tub - tlb) >= min_samples:
                if initial_state == samples[tlb]:
                    continue
                edge = 'rising' if samples[tlb] == 1 else 'falling'
                initial_state = samples[tlb]
                ts = t_prior + tlb
                events.append((edge, ts / fs))
        if events:
            target(events)
        t_prior += new_samples.shape[-1]
        prior_samples = samples[..., -min_samples:]


class Edges(EventInput):
    initial_state = d_(Int(0)).tag(metadata=True)
    debounce = d_(Int()).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        return edges(self.initial_state, self.debounce, self.fs, cb).send


################################################################################
# Epoch input types
################################################################################
class ExtractEpochs(EpochInput):

    added_queue = d_(Typed(deque, {}))
    removed_queue = d_(Typed(deque, {}))

    buffer_size = d_(Float(0)).tag(metadata=True)

    #: Defines the size of the epoch (if NaN, this is automatically drawn from
    #: the information provided by the queue).
    epoch_size = d_(Float(0)).tag(metadata=True)

    #: Defines the extra time period to capture beyond the epoch duration.
    poststim_time = d_(Float(0).tag(metadata=True))

    complete = Bool(False)

    #: The event that indicates beginning of an epoch. If left blank, the
    #: programmer is responsible for explicitly hooking up the event (e.g.,
    #: such as to a queue) or calling `.
    epoch_event = d_(Str())

    def mark_complete(self):
        self.complete = True

    def configure_callback(self):
        if np.isinf(self.epoch_size):
            m = f'ExtractEpochs {self.name} has an infinite epoch size'
            raise ValueError(m)
        cb = super().configure_callback()
        return extract_epochs(self.fs, self.added_queue, self.epoch_size,
                              self.poststim_time, self.buffer_size, cb,
                              self.mark_complete, self.removed_queue).send

    def _get_duration(self):
        return self.epoch_size + self.poststim_time

    # force change notification for duration
    def _observe_epoch_size(self, event):
        self.notify('duration', self.duration)

    def _observe_poststim_time(self, event):
        self.notify('duration', self.duration)


@coroutine
def reject_epochs(reject_threshold, mode, status, valid_target):
    if mode == 'absolute value':
        accept = lambda s: np.max(np.abs(s)) < reject_threshold
    elif mode == 'amplitude':
        accept = lambda s: np.ptp(s) < reject_threshold

    while True:
        epochs = (yield)
        # Check for valid epochs and send them if there are any
        valid = [e for e in epochs if accept(e)]
        if len(valid):
            valid_target(valid)

        def update():
            # Update the status. Must be wrapped in a deferred call to ensure
            # that the update occurs on the GUI thread.
            status.total += len(epochs)
            status.rejects += len(epochs) - len(valid)
            status.reject_percent = status.rejects / status.total * 100

        deferred_call(update)


class RejectEpochs(EpochInput):
    '''
    Rejects epochs whose amplitude exceeds a specified threshold.

    Attributes
    ----------
    threshold : float
        Reject threshold
    mode : {'absolute value', 'amplitude'}
        If absolute value, rejects epoch if the minimum or maximum exceeds the
        reject threshold. If amplitude, rejects epoch if the difference between
        the minimum and maximum exceeds the reject threshold.
    '''
    threshold = d_(Float()).tag(metadata=True)
    mode = d_(Enum('absolute value', 'amplitude')).tag(metadata=True)

    total = Int()
    rejects = Int()
    reject_percent = Float()

    def configure_callback(self):
        valid_cb = super().configure_callback()
        return reject_epochs(self.threshold, self.mode, self, valid_cb).send


@coroutine
def detrend(mode, target):
    kwargs = {'axis': -1, 'type': mode}
    while True:
        epochs = []
        for epoch in (yield):
            if mode is not None:
                epochs = xr.apply_ufunc(signal.detrend, epoch, kwargs=kwargs)
            epochs['attrs']['detrend_type'] = mode
            epochs.append(epoch)
        target(epochs)


class Detrend(EpochInput):
    '''
    Removes linear trend from epoch

    Attributes
    ----------
    mode : {None, 'linear', 'constant'}
        If None, this acts as a passthrough. If 'linear', the result of a
        linear least-squares fit is subtracted from the epoch. If 'constant',
        only the mean of the epoch is subtracted.
    '''
    mode = d_(Enum('constant', 'linear', None)).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        return detrend(self.mode, cb).send
