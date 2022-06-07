import logging

log = logging.getLogger(__name__)

from collections import deque
from functools import partial

import numpy as np
from scipy import signal

from atom.api import (Str, Float, Typed, Int, Property, Enum, Bool,
                      Callable, List, Tuple, set_default)
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_

from psiaudio.calibration import FlatCalibration

from psiaudio import pipeline
from psiaudio.util import db, dbi

from .channel import Channel

from psi.core.enaml.api import PSIContribution


class Input(PSIContribution):

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
        return pipeline.broadcast(*targets).send

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


class Transform(ContinuousInput):
    function = d_(Callable())

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.transform(self.function, cb).send


class Coroutine(Input):
    coroutine = d_(Callable())
    args = d_(Tuple())
    force_active = set_default(True)

    def configure_callback(self):
        cb = super().configure_callback()
        return self.coroutine(*self.args, cb).send


################################################################################
# Continuous input types
################################################################################
class CalibratedInput(Transform):
    '''
    Applies calibration to input

    Currently assumes that calibration is flat across frequency.

    If input is from a microphone and the microphone calibration transforms
    from Vrms to Pascals, then the output of this block will be in Pascals.
    '''
    def _default_function(self):
        sens = dbi(self.source.calibration.get_sens(1e3))
        return lambda x, s=sens: x * s

    def _get_calibration(self):
        # Input is now calibrated, and no additional transforms need to be
        # performed by downstream inputs.
        return FlatCalibration(sensitivity=0)


class RMS(ContinuousInput):

    #: Duration of window to calculate RMS for
    duration = d_(Float()).tag(metadata=True)

    def _get_fs(self):
        n = round(self.duration * self.source.fs)
        return self.source.fs / n

    def configure_callback(self):
        n = round(self.duration * self.source.fs)
        cb = super().configure_callback()
        return pipeline.rms(n, cb).send


class SPL(Transform):

    def _default_function(self):
        sens = self.calibration.get_sens(1e3)
        return lambda x, s=sens: db(x) + s


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

    Wn = Property().tag(metadata=True)

    def _get_Wn(self):
        if self.btype == 'lowpass':
            return self.f_lowpass
        elif self.btype == 'highpass':
            return self.f_highpass
        else:
            return (self.f_highpass, self.f_lowpass)

    def configure_callback(self):
        cb = super().configure_callback()
        if self.passthrough:
            return cb
        return pipeline.iirfilter(self.fs, self.N, self.Wn, None, None,
                                  self.btype, self.ftype, cb).send


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
        return pipeline.blocked(block_size, cb).send


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
        return pipeline.accumulate(self.n, self.axis, self.newaxis,
                                   self.status_cb, cb).send


class Capture(ContinuousInput):

    #: This is used internally by the CaptureManifest to notify Capture that
    #: new trials have been enqueued.
    queue = Typed(deque, ())

    #: The event that indicates beginning of capture. If left blank, the
    #: programmer is responsible for explicitly hooking up the event.
    start_event = d_(Str())

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.capture(self.fs, self.queue, cb).send


class Downsample(ContinuousInput):
    q = d_(Int()).tag(metadata=True)

    def _get_fs(self):
        return self.source.fs / self.q

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.downsample(self.q, cb).send


class Decimate(ContinuousInput):
    q = d_(Int()).tag(metadata=True)

    def _get_fs(self):
        return self.source.fs / self.q

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.decimate(self.q, cb).send


class Discard(ContinuousInput):
    duration = d_(Float()).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        samples = round(self.duration * self.fs)
        return pipeline.discard(samples, cb).send


class Threshold(Transform):

    threshold = d_(Float(0)).tag(metadata=True)

    def _default_function(self):
        return lambda x, t=self.threshold: x >= t


class Average(ContinuousInput):
    n = d_(Float()).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.average(self.n, cb).send


class Delay(ContinuousInput):
    # This can be set to account for things such as the AO and AI filter delays
    # on the 4461. For AO at 100 kHz, the output delay is ~0.48 msec. For the
    # AI at 25e-3, the input delay is 63 samples (divide this by the
    # acquisition rate).
    delay = d_(Float(0)).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        n = round(self.delay * self.fs)
        return pipeline.delay(n, cb).send


################################################################################
# Event input types
################################################################################
class Edges(EventInput):
    initial_state = d_(Int(0)).tag(metadata=True)
    debounce = d_(Int()).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.edges(self.initial_state, self.debounce, self.fs, cb).send


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
        return pipeline.extract_epochs(
            self.fs, self.added_queue, self.epoch_size, self.poststim_time,
            self.buffer_size, cb, self.mark_complete, self.removed_queue).send

    def _get_duration(self):
        return self.epoch_size + self.poststim_time

    # force change notification for duration
    def _observe_epoch_size(self, event):
        self.notify('duration', self.duration)

    def _observe_poststim_time(self, event):
        self.notify('duration', self.duration)


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

    def status_cb(self, n_total, n_accepted):
        def update():
            # Update the status. Must be wrapped in a deferred call to ensure
            # that the update occurs on the GUI thread.
            nonlocal self
            nonlocal n_total
            nonlocal n_accepted
            self.total += n_total
            self.rejects += n_total - n_accepted
            self.reject_percent = self.rejects / self.total * 100
        deferred_call(update)

    def configure_callback(self):
        valid_cb = super().configure_callback()
        return pipeline.reject_epochs(self.threshold, self.mode,
                                      self.status_cb, valid_cb).send



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
        return pipeline.detrend(self.mode, cb).send
