import logging
log = logging.getLogger(__name__)

from collections import namedtuple
from copy import copy
from functools import partial
from queue import Empty, Queue

import numpy as np
from scipy import signal

from atom.api import (Unicode, Float, Typed, Int, Property, Enum, Bool,
                      Callable, List)
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_
from ..util import coroutine
from .channel import Channel
from .calibration.util import db, dbi, patodb
from .device import Device
from .queue import AbstractSignalQueue

from psi.core.enaml.api import PSIContribution
from psi.controller.calibration import FlatCalibration


class InputData(np.ndarray):

    def __new__(cls, input_array, metadata=None):
        obj = np.asarray(input_array).view(cls)
        obj.metadata = metadata if metadata else {}
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)


def concatenate(input_data, axis=None):
    b = input_data[0]
    for d in input_data[1:]:
        if d.metadata != b.metadata:
            log.debug('%r vs %r', d.metadata, b.metadata)
            raise ValueError('Cannot combine InputData set')
    arrays = np.concatenate(input_data, axis=axis)
    return InputData(arrays, b.metadata)


@coroutine
def broadcast(*targets):
    while True:
        data = (yield)
        for target in targets:
            target(data)


class Input(PSIContribution):

    name = d_(Unicode()).tag(metadata=True)
    label = d_(Unicode()).tag(metadata=True)
    force_active = d_(Bool(False)).tag(metadata=True)

    source_name = d_(Unicode())
    source = d_(Typed(Declarative).tag(metadata=True), writable=False)
    channel = Property()
    engine = Property()

    fs = Property().tag(metadata=True)
    dtype = Property().tag(metadata=True)
    unit = Property().tag(metadata=True)
    calibration = Property()
    active = Property()

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

    def configure_callback(self):
        targets = [i.configure_callback() for i in self.inputs if i.active]
        log.debug('Configured callback for %s with %d targets', self.name, len(targets))
        if len(targets) == 1:
            return targets[0]
        # If we have more than one target, need to add a broadcaster
        return broadcast(*targets).send

    def add_callback(self, cb):
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
    while True:
        data = (yield)
        target(data/sens)


class CalibratedInput(ContinuousInput):

    def _get_calibration(self):
        return FlatCalibration(0)

    def configure_callback(self):
        cb = super().configure_callback()
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
            result = np.mean(data[..., :n]**2, axis=0)**0.5
            target(result[np.newaxis])
            data = data[..., n:]


class RMS(ContinuousInput):

    duration = d_(Float()).tag(metadata=True)

    def _get_fs(self):
        n = round(self.duration*self.source.fs)
        return self.source.fs/n

    def configure_callback(self):
        n = round(self.duration*self.source.fs)
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
    zo = zi*y[0]

    while True:
        y, zo = signal.lfilter(b, a, y, zi=zo)
        target(y)
        y = (yield)


class IIRFilter(ContinuousInput):

    # Allows user to deactivate the filter entirely during configuration if
    # desired. Ideally we could just remove it from the graph, but it seems a
    # bit tricky to do so as some other components may be looking for the
    # output of this block.
    passthrough = d_(Bool(False)).tag(metadata=True)

    N = d_(Int(1)).tag(metadata=True)
    btype = d_(Enum('bandpass', 'lowpass', 'highpass', 'bandstop')).tag(metadata=True)
    ftype = d_(Enum('butter', 'cheby1', 'cheby2', 'ellip', 'bessel')).tag(metadata=True)
    f_highpass = d_(Float()).tag(metadata=True)
    f_lowpass = d_(Float()).tag(metadata=True)
    wn = Property()

    def _get_wn(self):
        if self.btype == 'lowpass':
            log.debug('Lowpass at %r (fs=%r)', self.f_lowpass, self.fs)
            return self.f_lowpass/(0.5*self.fs)
        elif self.btype == 'highpass':
            log.debug('Highpass at %r (fs=%r)', self.f_lowpass, self.fs)
            return self.f_highpass/(0.5*self.fs)
        else:
            log.debug('Bandpass %r to %r (fs=%r)', self.f_highpass,
                      self.f_lowpass, self.fs)
            return (self.f_highpass/(0.5*self.fs),
                    self.f_lowpass/(0.5*self.fs))

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
                target(merged[..., :block_size])
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
        block_size = round(self.duration*self.fs)
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
    t0 = 0
    t_start = None  # track start of capture
    t_next = None
    active = False

    while True:
        # Wait for new data to come in
        data = (yield)

        try:
            # We've recieved a new command. The command will either be None
            # (i.e., no more acquisition for a bit) or a floating-point value
            # (indicating when next acquisition should begin).
            t_next = queue.get(block=False)
            if t_next is not None:
                log.debug('Starting capture at %f', t_next)
                t_next = round(t_next*fs)
                t_start = t_next
                target(Ellipsis)
            elif t_next is None:
                log.debug('Ending capture')
            elif t_next < t0:
                raise SystemError('Data lost')
        except Empty:
            pass

        if (t_next is not None) and (t_next >= t0):
            i = t_next-t0
            if i < data.shape[-1]:
                d = data[i:]
                d.metadata['capture'] = t_start
                target(d)
                t_next += d.shape[-1]

        t0 += data.shape[-1]


class Capture(ContinuousInput):

    queue = Typed(Queue)

    def configure_callback(self):
        self.queue = Queue()
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
        return self.source.fs/self.q

    def configure_callback(self):
        cb = super().configure_callback()
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

    def configure_callback(self):
        cb = super().configure_callback()
        return decimate(self.q, cb).send


@coroutine
def discard(discard_samples, cb):
    discarded = discard_samples
    while True:
        samples = (yield)
        if samples is Ellipsis:
            to_discard = discarded
            cb(samples)
            continue

        if discard_samples == 0:
            cb(samples)
        elif samples.shape[-1] <= discard_samples:
            discard_samples -= samples.shape[-1]
        elif samples.shape[-1] > discard_samples:
            s = samples[..., discard_samples:]
            discard_samples -= s.shape[-1]
            cb(s)


class Discard(ContinuousInput):

    duration = d_(Float()).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        samples = round(self.duration*self.fs)
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
            s = [Ellipsis]*data.ndim
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
        n = int(self.delay * self.fs)
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
            if (tub-tlb) >= min_samples:
                if initial_state == samples[tlb]:
                    continue
                edge = 'rising' if samples[tlb] == 1 else 'falling'
                initial_state = samples[tlb]
                ts = t_prior + tlb
                events.append((edge, ts/fs))
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
            log.warn(m, epoch_samples, epoch_t0)
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
def extract_epochs(fs, queue, epoch_size, poststim_time, buffer_size, target,
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
        prior_samples.append((tlb, data))

        # Send the data to each coroutine. If a StopIteration occurs, this means
        # that the epoch has successfully been acquired and has been sent to the
        # callback and we can remove it. Need to operate on a copy of list since
        # it's bad form to modify a list in-place.
        for epoch_coroutine in epoch_coroutines[:]:
            try:
                epoch_coroutine.send((tlb, data))
            except StopIteration:
                epoch_coroutines.remove(epoch_coroutine)

        # Check to see if more epochs have been requested. Information will be
        # provided in seconds, but we need to convert this to number of
        # samples.
        while trial_index < len(queue.uploaded):
            info = queue.uploaded[trial_index].copy()
            trial_index += 1

            # Figure out how many samples to capture for that epoch
            t0 = round(info['t0'] * fs)
            info['poststim_time'] = poststim_time
            if epoch_size:
                info['epoch_size'] = epoch_size
                total_epoch_size = epoch_size + poststim_time
            else:
                info['epoch_size'] = info['duration']
                total_epoch_size = info['duration'] + poststim_time

            info['total_epoch_size'] = total_epoch_size
            epoch_samples = round(total_epoch_size * fs)

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


class ExtractEpochs(EpochInput):

    queue = d_(Typed(AbstractSignalQueue))
    buffer_size = d_(Float(0)).tag(metadata=True)

    # Defines the size of the epoch (if 0, this is automatically drawn from
    # the information provided by the queue).
    epoch_size = d_(Float(0)).tag(metadata=True)

    # Defines the extra time period to capture beyond the epoch duration.
    poststim_time = d_(Float(0).tag(metadata=True))

    complete = Bool(False)

    def mark_complete(self):
        self.complete = True

    def configure_callback(self):
        if self.epoch_size <= 0:
            self.epoch_size = self.queue.get_max_duration()
        if not np.isfinite(self.epoch_size):
                raise SystemError('Cannot have an infinite epoch size')
        cb = super().configure_callback()
        return extract_epochs(self.fs, self.queue, self.epoch_size,
                              self.poststim_time, self.buffer_size, cb,
                              self.mark_complete).send

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
        valid = [e for e in epochs if accept(e['signal'])]
        if len(valid):
            valid_target(valid)

        def update():
            # Update the status. Must be wrapped in a deferred call to ensure
            # that the update occurs on the GUI thread.
            status.total += len(epochs)
            status.rejects += len(epochs)-len(valid)
            status.reject_ratio = status.rejects / status.total

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
    reject_ratio = Float()

    def configure_callback(self):
        valid_cb = super().configure_callback()
        return reject_epochs(self.threshold, self.mode, self, valid_cb).send


@coroutine
def detrend(mode, target):
    if mode is None:
        do_detrend = lambda x: x
    else:
        do_detrend = partial(signal.detrend, type=mode)
    while True:
        epochs = []
        for epoch in (yield):
            epoch = {
                'signal': do_detrend(epoch['signal']),
                'info': epoch['info']
            }
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
    mode = d_(Enum('constant', 'linear', None))

    def configure_callback(self):
        cb = super().configure_callback()
        return detrend(self.mode, cb).send
