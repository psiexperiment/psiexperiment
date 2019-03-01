import logging
log = logging.getLogger(__name__)

from types import GeneratorType
from functools import partial

import numpy as np

from atom.api import (Unicode, Enum, Typed, Property, Float, Int, Bool, List)

import enaml
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Extension

from ..util import coroutine, SignalBuffer
from .queue import AbstractSignalQueue

from psi.core.enaml.api import PSIContribution

import time


class Synchronized(PSIContribution):

    outputs = Property()
    engines = Property()

    def _get_outputs(self):
        return self.children

    def _get_engines(self):
        return set(o.engine for o in self.outputs)


class Output(PSIContribution):

    name = d_(Unicode()).tag(metadata=True)
    label = d_(Unicode()).tag(metadata=True)

    target_name = d_(Unicode())
    target = Typed(Declarative).tag(metadata=True)
    channel = Property()
    engine = Property()

    # These two are defined as properties because it's theoretically possible
    # for the output to transform these (e.g., an output could upsample
    # children or "equalize" something before passing it along).
    fs = Property().tag(metadata=True)
    calibration = Property().tag(metadata=True)

    # TODO: clean this up. it's sort of hackish.
    token = d_(Typed(Declarative))

    # Can the user configure properties (such as the token) via the GUI?
    configurable = d_(Bool(True))

    def _get_engine(self):
        return self.channel.engine

    def _get_channel(self):
        from .channel import Channel
        target = self.target
        while True:
            if isinstance(target, Channel):
                return target
            else:
                target = target.target

    def _get_fs(self):
        return self.channel.fs

    def _get_calibration(self):
        return self.channel.calibration

    def is_ready(self):
        raise NotImplementedError


class BufferedOutput(Output):

    dtype = Unicode('double')
    buffer_size = Property()
    active = Bool(False)
    source = Typed(object)

    _buffer = Typed(SignalBuffer)
    _offset = Int(0)

    def _get_buffer_size(self):
        return self.channel.buffer_size

    def _default__buffer(self):
        print(self.name, self.fs, self.buffer_size)
        return SignalBuffer(self.fs, self.buffer_size, 0, self.dtype)

    def get_samples(self, offset, samples, out):
        lb = offset
        ub = offset + samples
        buffered_lb = self._buffer.get_samples_lb()
        buffered_ub = self._buffer.get_samples_ub()

        log.trace('Getting %d samples from %d to %d for %s', samples, lb, ub,
                  self.name)
        log.trace('Buffer has %d to %d for %s', buffered_lb, buffered_ub,
                  self.name)

        if lb > buffered_ub:
            # This breaks an implicit software contract.
            raise SystemError('Mismatch between offsets')
        elif lb == buffered_ub:
            log.trace('Generating new data')
            pass
        elif lb >= buffered_lb and ub <= buffered_ub:
            print(self.name, lb, buffered_lb, ub, buffered_ub)
            log.trace('Extracting from buffer')
            out[:] = self._buffer.get_range_samples(lb, ub)
            samples = 0
        elif lb >= buffered_lb and ub > buffered_ub:
            log.trace('Extracting from buffer and generating new data')
            b = self._buffer.get_range_samples(lb)
            s = b.shape[-1]
            out[:s] = b
            samples -= s

        # Generate new samples
        if samples > 0:
            data = self.get_next_samples(samples)
            self._buffer.append_data(data)
            out[-samples:] = data

    def get_next_samples(self, samples):
        raise NotImplementedError

    def activate(self, offset):
        log.debug('Activating %s at %d', self.name, offset)
        self.active = True
        self._offset = offset
        self._buffer.invalidate_samples(offset)
        log.debug(self._buffer.get_samples_ub())

    def deactivate(self, offset):
        log.debug('Deactivating %s at %d', self.name, offset)
        self.active = False
        self.source = None
        self._buffer.invalidate_samples(offset)

    def is_ready(self):
        return self.source is not None

    def get_duration(self):
        return self.source.get_duration()


class EpochOutput(BufferedOutput):

    def get_next_samples(self, samples):
        log.trace('Getting %d samples for %s', samples, self.name)
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
    auto_decrement = d_(Bool(False))
    complete_cb = Typed(object)
    queue = Property()

    def _get_queue(self):
        return self.source

    def _set_queue(self, value):
        self.source = value

    def _observe_target(self, event):
        self._update_queue()

    def _observe_source(self, event):
        self._update_queue()

    def _update_queue(self):
        if self.queue is not None and self.target is not None:
            self.queue.set_fs(self.fs)

    def get_next_samples(self, samples):
        if self.active:
            waveform, empty = self.queue.pop_buffer(samples, self.auto_decrement)
            if empty and self.complete_cb is not None:
                log.debug('Queue empty. Calling complete callback.')
                deferred_call(self.complete_cb)
                self.active = False
        else:
            waveform = np.zeros(samples, dtype=np.double)
        return waveform

    def add_setting(self, setting, averages=None, iti_duration=None):
        with enaml.imports():
            from .output_manifest import initialize_factory

        # Make a copy to ensure that we don't accidentally modify in-place
        context = setting.copy()

        if averages is None:
            averages = context.pop(f'{self.name}_averages')
        if iti_duration is None:
            iti_duration = context.pop(f'{self.name}_iti_duration')

        # Somewhat surprisingly it appears to be faster to use factories in the
        # queue rather than creating the waveforms for ABR tone pips, even for
        # very short signal durations.
        context['fs'] = self.fs
        context['calibration'] = self.calibration

        # I'm not in love with this since it requires hooking into the
        # manifest system.
        factory = initialize_factory(self, self.token, context)
        duration = factory.get_duration()
        self.queue.append(factory, averages, iti_duration, duration, setting)

    def activate(self, offset):
        super().activate(offset)
        self.queue.set_t0(offset/self.fs)

    def get_duration(self):
        # TODO: add a method to get actual duration from queue.
        return np.inf


class SelectorQueuedEpochOutput(QueuedEpochOutput):

    selector_name = d_(Unicode())


class ContinuousOutput(BufferedOutput):

    def get_next_samples(self, samples):
        if self.active:
            return self.source.next(samples)
        else:
            return np.zeros(samples, dtype=np.double)


class DigitalOutput(Output):
    pass


class Trigger(DigitalOutput):

    duration = d_(Float(0.1))

    def fire(self):
        self.engine.fire_sw_do(self.channel.name, duration=self.duration)


class Toggle(DigitalOutput):

    state = Bool(False)

    def _observe_state(self, event):
        try:
            # TODO: Fixme
            self.engine.set_sw_do(self.channel.name, event['value'])
        except:
            pass

    def set_high(self):
        self.state = True

    def set_low(self):
        self.state = False
