import logging
log = logging.getLogger(__name__)

from types import GeneratorType
from functools import partial

import numpy as np

from atom.api import (Str, Enum, Event, Typed, Property, Float, Int, Bool,
                      List)

import enaml
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Extension

from psiaudio.queue import AbstractSignalQueue

from psi.util import SignalBuffer
from psi.core.enaml.api import PSIContribution

import time


class Synchronized(PSIContribution):

    outputs = Property()
    engines = Property()

    def _get_outputs(self):
        return self.children

    def _get_engines(self):
        return set(o.engine for o in self.outputs)


class BaseOutput(PSIContribution):

    name = d_(Str()).tag(metadata=True)
    label = d_(Str()).tag(metadata=True)

    target_name = d_(Str()).tag(metadata=True)
    target = d_(Typed(Declarative).tag(metadata=True), writable=False)
    channel = Property().tag(metadata=True)
    engine = Property().tag(metadata=True)

    def _default_label(self):
        return self.name

    def _get_engine(self):
        if self.channel is None:
            return None
        else:
            return self.channel.engine

    def _get_channel(self):
        from .channel import Channel
        target = self.target
        while True:
            if target is None:
                return None
            elif isinstance(target, Channel):
                return target
            else:
                target = target.target

    def is_ready(self):
        raise NotImplementedError


class Output(BaseOutput):

    # These two are defined as properties because it's theoretically possible
    # for the output to transform these (e.g., an output could upsample
    # children or "equalize" something before passing it along).
    fs = Property().tag(metadata=True)
    calibration = Property().tag(metadata=True)
    filter_delay = Property().tag(metadata=True)

    # TODO: clean this up. it's sort of hackish.
    token = d_(Typed(Declarative)).tag(metadata=True)

    # Can the user configure properties (such as the token) via the GUI?
    configurable = d_(Bool(True))

    callbacks = List()

    def connect(self, cb):
        self.callbacks.append(cb)

    def notify(self, data):
        if not self.callbacks:
            return
        # Correct for filter delay
        d = data.copy()
        d['t0'] += self.filter_delay
        for cb in self.callbacks:
            cb(d)

    def _get_filter_delay(self):
        return self.target.filter_delay

    def _get_fs(self):
        return self.channel.fs

    def _get_calibration(self):
        return self.channel.calibration


class BufferedOutput(Output):

    dtype = Str('double').tag(metadata=True)
    buffer_size = Property().tag(metadata=True)
    active = Bool(False).tag(metadata=True)
    paused = Bool(False)

    #: This is managed by the manifest
    source = Typed(object).tag(metadata=True)
    source_md = Dict()

    _buffer = Typed(SignalBuffer)
    _offset = Int(0)

    def _get_buffer_size(self):
        return self.channel.buffer_size

    def _default__buffer(self):
        return SignalBuffer(self.fs, self.buffer_size, 0, self.dtype)

    def get_samples(self, offset, samples, out):
        lb = offset
        ub = offset + samples
        buffered_lb = self._buffer.get_samples_lb()
        buffered_ub = self._buffer.get_samples_ub()
        if lb > buffered_ub:
            # This breaks an implicit software contract.
            m = 'Mismatch between offsets. ' \
                f'Requested {lb} but only buffered up to {buffered_lb}'
            raise SystemError(m)
        elif lb == buffered_ub:
            log.trace('Generating new data')
            pass
        elif lb >= buffered_lb and ub <= buffered_ub:
            out[:] = self._buffer.get_range_samples(lb, ub)
            samples = 0
            offset = ub
        elif lb >= buffered_lb and ub > buffered_ub:
            b = self._buffer.get_range_samples(lb)
            s = b.shape[-1]
            out[:s] = b
            samples -= s
            offset += s

        # Don't generate new samples if occuring before activation.
        if (samples > 0) and (offset < self._offset):
            s = min(self._offset-offset, samples)
            data = np.zeros(s)
            self._buffer.append_data(data)
            if (samples == s):
                out[-samples:] = data
            else:
                out[-samples:-samples+s] = data
            samples -= s
            offset += s

        # Generate new samples
        if samples > 0:
            data = self.get_next_samples(samples)
            self._buffer.append_data(data)
            out[-samples:] = data

    def get_next_samples(self, samples):
        raise NotImplementedError

    def activate(self, offset):
        log.trace('Activating %s at %d', self.name, offset)
        self.active = True
        self.paused = False
        self._offset = offset
        self._buffer.invalidate_samples(offset)

    def deactivate(self, offset):
        log.trace('Deactivating %s at %d', self.name, offset)
        self.active = False
        self.source = None
        self._buffer.invalidate_samples(offset)

    def is_ready(self):
        return self.source is not None

    def get_duration(self):
        return self.source.get_duration()

    def rebuffer(self, time, delay=0):
        offset = round(time * self.fs)
        self._buffer.invalidate_samples(offset)
        self.engine.update_hw_ao(self.channel.name, offset)

    def pause(self, time):
        self.paused = True
        self.rebuffer(time)

    def resume(self, time):
        self.paused = False
        self.rebuffer(time)


class EpochOutput(BufferedOutput):

    def get_next_samples(self, samples):
        if self.active:
            buffered_ub = self._buffer.get_samples_ub()

            # Pad with zero
            zero_padding = max(self._offset-buffered_ub, 0)
            zero_padding = min(zero_padding, samples)
            waveform_samples = samples - zero_padding

            waveforms = []
            if zero_padding:
                w = np.zeros(zero_padding, dtype=self.dtype)
                waveforms.append(w)
            if waveform_samples:
                w = self.source.next(waveform_samples)
                waveforms.append(w)
            if self.source.is_complete():
                self.deactivate(self._buffer.get_samples_ub())
            waveform = np.concatenate(waveforms, axis=-1)
        else:
            waveform = np.zeros(samples, dtype=self.dtype)
        return waveform


class QueuedEpochOutput(BufferedOutput):

    queue = d_(Typed(AbstractSignalQueue))
    #: Automatically decrement the number of trials left to present? Set to
    #: False if you plan to handle this yourself (e.g., in the case of artifact
    #: reject).
    auto_decrement = d_(Bool(False)).tag(metadata=True)


    complete = d_(Event(), writable=False)
    paused = Bool(False)

    removed_callbacks = List()

    def connect(self, cb, event='added'):
        if event == 'added':
            self.callbacks.append(cb)
        elif event == 'removed':
            self.removed_callbacks.append(cb)

    def notify_removed(self, data):
        if not self.removed_callbacks:
            return
        d = data.copy()
        d['t0'] += self.filter_delay
        for cb in self.removed_callbacks:
            cb(d)

    def rebuffer(self, time, delay=0):
        self.queue.cancel(time, delay)
        self.queue.rewind_samples(time)
        super().rebuffer(time, delay)

    def pause(self, time):
        self.queue.pause(time)
        super().pause(time)

    def resume(self, time, delay=0):
        self.queue.resume(time)
        super().resume(time)

    def _observe_queue(self, event):
        self.source = self.queue
        self._update_queue()

    def _observe_target(self, event):
        self._update_queue()

    def _update_queue(self):
        if self.queue is not None and self.target is not None:
            self.queue.set_fs(self.fs)
            self.queue.connect(self.notify, 'added')
            self.queue.connect(self.notify_removed, 'removed')

    def get_next_samples(self, samples):
        if self.active:
            waveform = self.queue.pop_buffer(samples, self.auto_decrement)
            if self.queue.is_empty():
                self.complete = True
                self.active = False
                log.debug('Queue empty. Output %s no longer active.', self.name)
        else:
            waveform = np.zeros(samples, dtype=np.double)
        return waveform

    def add_setting(self, setting, averages=None, iti_duration=None,
                    total_duration=None):

        if iti_duration is not None and total_duration is not None:
            raise ValueError('Cannot specify both iti_duration and total_duration')
        elif iti_duration is None and total_duration is None:
            raise ValueError('must specify either iti_duration or total_duration')

        with enaml.imports():
            # TODO: HACK ALERT!
            from .output_manifest import initialize_factory

        # Make a copy to ensure that we don't accidentally modify in-place
        context = setting.copy()

        # I'm not in love with this since it requires hooking into the
        # manifest system.
        factory = initialize_factory(self, self.token, context)
        duration = factory.get_duration()
        if total_duration is not None:
            iti_duration = total_duration - duration
        return self.queue.append(factory, averages, iti_duration, duration,
                                 setting.copy())

    def activate(self, offset):
        log.debug('Activating output at %d', offset)
        super().activate(offset)
        self.queue.set_t0(offset/self.fs)

    def get_duration(self):
        # TODO: add a method to get actual duration from queue.
        return np.inf


class SelectorQueuedEpochOutput(QueuedEpochOutput):

    selector_name = d_(Str('default')).tag(metadata=True)


class ContinuousOutput(BufferedOutput):

    def get_next_samples(self, samples):
        if not self.paused and self.active:
            return self.source.next(samples)
        else:
            return np.zeros(samples, dtype=np.double)


class DigitalOutput(BaseOutput):

    pass


class Trigger(DigitalOutput):

    #: Total number of triggers.
    total_fired = d_(Int(0), writable=False)

    #: Specifies the default duration for the trigger. This can be overridden
    #: by specifying a duration when calling `fire`.
    duration = d_(Float(0.1))

    def fire(self, duration=None):
        if duration is None:
            duration = self.duration
        if self.engine.configured:
            self.engine.fire_sw_do(self.channel.name, duration=duration)
            self.total_fired += 1


class Toggle(DigitalOutput):

    state = Bool(False)

    def _observe_state(self, event):
        if self.engine is not None and self.engine.configured:
            self.engine.set_sw_do(self.channel.name, event['value'])

    def set_high(self):
        self.state = True

    def set_low(self):
        self.state = False
