'''
Developer tips
--------------

Sampling rate
.............

When providing sampling rate to pipeline coroutines, you usually want the
sampling rate of the input data (``self.source.fs``) rather than the sampling
rate of the coroutine output (``self.fs``).
'''
import logging

log = logging.getLogger(__name__)

from collections import deque
from functools import partial

import numpy as np
from scipy import signal

from atom.api import (
    Bool, Callable, Dict, Enum, Float, Int, List, Property, set_default, Str,
    Tuple, Typed, Value
)
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_

from psiaudio.calibration import FlatCalibration

from psiaudio import pipeline
from psiaudio.util import db, dbi

from .channel import Channel

from psi.core.enaml.api import PSIContribution


class Input(PSIContribution):
    '''
    Base class for an input that receives data from a single parent
    :py:attr:`~source` and performs an operation on the data before passing the
    transformed data along to one or more children :py:attr:`~inputs`.
    '''

    #: If False, the controller plugin will attempt to determine whether the
    #: outputs of the processing chain are used by other plugins (e.g., the
    #: input is being plotted or saved). If the output does not appear to be
    #: used, the controller will skip the input.
    force_active = d_(Bool(False)).tag(metadata=True)

    source_name = d_(Str()).tag(metadata=True)

    #: Source that input receives data from.
    source = d_(Typed(Declarative).tag(metadata=True), writable=False)

    #: Channel that input receives data from. This is the initial source
    #: feeding data into the processing chain that leads to this input.
    #: There may be intermediate steps in the chain that perform some
    #: manipulations of the data (e.g., decimation, etc.) before this input
    #: receives the data. Use :py:attr:`~source` to get the previous step in
    #: the processing chain.
    channel = Property().tag(metadata=True)

    #: Engine that input receives data from
    engine = Property().tag(metadata=True)

    #: Number of channels
    n_channels = Property().tag(metadata=True)

    #: List mapping channel index to label.
    channel_labels = Property().tag(metadata=True)

    #: Sampling rate of input.
    fs = Property().tag(metadata=True)

    #: Datatype of samples (e.g., int, float, double)
    dtype = Property().tag(metadata=True)

    unit = Property().tag(metadata=True)
    calibration = Property().tag(metadata=True)

    #: Is the input active (i.e., receiving data and performing operations)?
    #: See :py:attr:`~force_active` for more detail.
    active = Property().tag(metadata=True)

    #: List of children in the processing chain to pass transformed data (i.e.,
    #: the output of this block). This input is the :py:attr:`~source` for the
    #: inputs in the list.
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

    def _get_n_channels(self):
        return self.source.n_channels

    def _get_channel_labels(self):
        return self.source.channel_labels

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


class MCReference(ContinuousInput):

    reference = d_(Enum('all', 'raw', 'FCz', 'MC1', 'MC2', 'MC1+MC2'))

    def configure_callback(self):
        cb = super().configure_callback()
        m = create_diff_matrix(self.n_channels, self.reference,
                               self.channel_labels)
        return pipeline.mc_reference(m, cb).send


class MCSelect(ContinuousInput):

    selected_channel = d_(Value())

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.mc_select(self.selected_channel,
                                  self.source.channel_labels, cb).send

    def _get_channel_labels(self):
        return self.selected_channel


class RMS(ContinuousInput):

    #: Duration of window to calculate RMS for
    duration = d_(Float()).tag(metadata=True)

    def _get_fs(self):
        n = round(self.duration * self.source.fs)
        return self.source.fs / n

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.rms(self.source.fs, self.duration, cb).send


class SPL(Transform):

    def _default_function(self):
        sens = self.calibration.get_sens(1e3)
        return lambda x, s=sens: db(x) + s


class IIRFilter(ContinuousInput):
    '''
    Apply an IIR filter to the data
    '''
    #: Allows user to deactivate the filter entirely during configuration if
    #: desired. Ideally we could just remove it from the processing chain, but
    #: it seems a bit tricky to do so as some other components may be looking
    #: for the output of this block.
    passthrough = d_(Bool(False)).tag(metadata=True)

    #: Filter order
    N = d_(Int(1)).tag(metadata=True)

    #: Filter pasband type
    btype = d_(Enum('bandpass', 'lowpass', 'highpass', 'bandstop')).tag(
        metadata=True)

    #: Filter type
    ftype = d_(Enum('butter', 'cheby1', 'cheby2', 'ellip', 'bessel')).tag(
        metadata=True)

    #: Highpass cutoff frequency (Hz)
    f_highpass = d_(Float()).tag(metadata=True)

    #: Lowpass cutoff frequency (Hz)
    f_lowpass = d_(Float()).tag(metadata=True)

    #: Filter passband (automatically adjusted based on :py:attr:`~btype`,
    #: :py:attr:`~f_highpass`, and :py:attr:`~f_lowpass`).
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

    #: Duration, in seconds, of each chunk. This is rounded to the nearest
    #: integer number of samples and each block will always have the exact same
    #: number of samples.
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

    #: Decimation factor (i.e., extract every ``q`` samples).
    q = d_(Int()).tag(metadata=True)

    def _get_fs(self):
        return self.source.fs / self.q

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.downsample(self.q, cb).send


class _Decimate(ContinuousInput):
    '''
    Base class for two approaches to decimation (either specifying the
    downsampling factor directly or the desired sampling rate).
    '''
    def _get_fs(self):
        return self.source.fs / self.q

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.decimate(self.q, cb).send


class Decimate(_Decimate):
    '''
    Decimate the acquired data to every q samples.

    Applys a chebyshev filter prior to decimation. Properly handles filter
    history.
    '''
    q = d_(Int()).tag(metadata=True)


class DecimateTo(_Decimate):
    '''
    Decimate the acquired data as close to the target sampling rate as
    possible.

    Applies a chebyshev filter prior to decimation. Properly handles filter
    history.
    '''
    #: Target sampling rate. Since decimation applies a lowpass filter prior to
    #: taking every q samples, the actual sampling rate will be an integer
    #: divisor of the source sampling rate. The actual sampling rate, fs, will
    #: never be less than target_fs.
    target_fs = d_(Float()).tag(metadata=True)

    #: Computed decimation factor.
    q = Property().tag(metadata=True)

    def _get_q(self):
        return int(np.floor(self.source.fs / self.target_fs))


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


class AutoThreshold(ContinuousInput):
    '''
    Automatically threshold data as `n` standard deviations computed over
    a window that is `baseline` seconds long.
    '''
    #: Number of standard deviations to set threshold at
    n = d_(Int(4))

    #: Duration, in seconds, to calculate threshold over.
    baseline = d_(Float(30))

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.auto_th(self.n, self.baseline, cb,
                                fs=self.parent.fs).send


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


class Bitmask(Transform):

    bit = d_(Int(0))

    def _default_function(self):
        return lambda x, b=self.bit: ((x >> b) & 1).astype('bool')


################################################################################
# Event input types
################################################################################
class Edges(EventInput):

    initial_state = d_(Int(0)).tag(metadata=True)

    #: Minimum number of samples required before a change is registered. This
    #: is effectively a means of "debouncing" the signal. The change time is
    #: the time at which the change occurred, not the time at which the minimum
    #: number of samples criterion was met.
    debounce = d_(Int(2)).tag(metadata=True)

    #: Edges to detect.
    detect = d_(Enum('rising', 'falling', 'both')).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.edges(min_samples=self.debounce,
                              initial_state=self.initial_state, target=cb,
                              fs=self.fs, detect=self.detect).send


class EventsToInfo(EventInput):

    trigger_edge = d_(Enum('rising', 'falling'))
    base_info = d_(Dict())

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.events_to_info(self.trigger_edge, self.base_info,
                                       cb).send


class EventRate(EventInput):
    '''
    Calculate the rate at which events occur using a sliding temporal window.
    '''

    #: Size of window, in seconds, to calculate event rate over.
    block_size = d_(Float(1)).tag(metadata=True)

    #: Increment, in seconds, to advance window before calculating next event
    #: rate.
    block_step = d_(Float(0.25)).tag(metadata=True)

    #: Increment, in samples, to advance window before calculating next event
    #: rate. Automatically calculated from input sampling rate and block_size.
    block_size_samples = Property().tag(metadata=True)

    #: Increment, in samples, to advance window before calculating next event
    #: rate. Automatically calculated from input sampling rate and block_step.
    block_step_samples = Property().tag(metadata=True)

    def _get_block_step_samples(self):
        return int(np.round(self.block_step * self.parent.fs))

    def _get_block_size_samples(self):
        return int(np.round(self.block_size * self.parent.fs))

    def _get_fs(self):
        return self.parent.fs / self.block_step_samples

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.event_rate(self.block_size_samples,
                                   self.block_step_samples, cb).send


################################################################################
# Epoch input types
################################################################################
class ExtractEpochs(EpochInput):

    added_queue = d_(Typed(deque, {}))
    removed_queue = d_(Typed(deque, {}))

    #: Duration to buffer (allowing for lookback captures where we belatedly
    #: notify the coroutine that we wish to capture an epoch).
    buffer_size = d_(Float(0)).tag(metadata=True)

    #: Defines the size of the epoch (if NaN, this is automatically drawn from
    #: the information provided by the queue).
    epoch_size = d_(Float(0)).tag(metadata=True)

    #: Defines the extra time period to capture beyond the epoch duration.
    poststim_time = d_(Float(0).tag(metadata=True))

    #: Defines the extra time period to capture before the epoch begins
    prestim_time = d_(Float(0).tag(metadata=True))

    #: Flag indicating that all queued epochs have been captured.
    complete = Bool(False)

    #: The event that indicates beginning of an epoch. If left blank, the
    #: programmer is responsible for explicitly hooking up the event (e.g.,
    #: such as to a queue).
    epoch_event = d_(Str())

    def mark_complete(self):
        self.complete = True

    def configure_callback(self):
        if np.isinf(self.epoch_size):
            m = f'ExtractEpochs {self.name} has an infinite epoch size'
            raise ValueError(m)
        cb = super().configure_callback()
        return pipeline.extract_epochs(
            fs=self.fs, queue=self.added_queue, epoch_size=self.epoch_size,
            buffer_size=self.buffer_size, target=cb,
            empty_queue_cb=self.mark_complete,
            removed_queue=self.removed_queue, prestim_time=self.prestim_time,
            poststim_time=self.poststim_time).send

    def _get_duration(self):
        return self.epoch_size + self.poststim_time + self.prestim_time

    # force change notification for duration
    def _observe_epoch_size(self, event):
        self.notify('duration', self.duration)

    # force change notification for poststim time
    def _observe_poststim_time(self, event):
        self.notify('duration', self.duration)


class RejectEpochs(EpochInput):
    '''
    Rejects epochs whose amplitude exceeds a specified threshold.
    '''
    #: Reject threshold
    threshold = d_(Float()).tag(metadata=True)

    #: If `'absolute value'`, rejects epoch if the minimum or maximum exceeds
    #: the reject threshold. If `'amplitude'`, rejects epoch if the difference
    #: between the minimum and maximum exceeds the reject threshold.
    mode = d_(Enum('absolute value', 'amplitude')).tag(metadata=True)

    #: Total number of epochs seen (both accepted and rejected).
    total = Int()

    #: Number of epochs rejected.
    rejects = Int()

    #: Percent of epochs rejected.
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
    '''

    #: If None, this acts as a passthrough. If `'linear'`, the result of a
    #: linear least-squares fit is subtracted from the epoch. If `'constant'`,
    #: only the mean of the epoch is subtracted.
    mode = d_(Enum('constant', 'linear', None)).tag(metadata=True)

    def configure_callback(self):
        cb = super().configure_callback()
        return pipeline.detrend(self.mode, cb).send
