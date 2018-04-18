import logging
log = logging.getLogger(__name__)

from types import GeneratorType
from functools import partial

import numpy as np

from atom.api import (Unicode, Enum, Typed, Property, Float, Int, Bool, List)

from enaml.application import deferred_call
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Extension

from ..util import coroutine, SignalBuffer
from .queue import AbstractSignalQueue

from psi.core.enaml.api import PSIContribution
from psi.token.primitives import Waveform

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
    filter_delay = Property().tag(metadata=True)

    # TODO: clean this up. it's sort of hackish.
    token_name = d_(Unicode())
    token = Typed(Declarative)

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

    def _get_filter_delay(self):
        return self.channel.filter_delay

    def _get_calibration(self):
        return self.channel.calibration



class AnalogOutput(Output):

    buffer_size = Property()

    _buffer = Typed(SignalBuffer)
    _offset = Int(0)

    def _get_buffer_size(self):
        return self.channel.buffer_size

    def _default__buffer(self):
        return SignalBuffer(self.fs, self.buffer_size, 0)

    def get_samples(self, offset, samples, out):
        lb = offset
        ub = offset + samples
        buffered_lb = self._buffer.get_samples_lb()
        buffered_ub = self._buffer.get_samples_ub()

        if lb > buffered_ub:
            # This breaks an implicit software contract.
            raise SystemError('Mismatch between offsets')
        elif lb == buffered_ub:
            log.debug('Generating new data')
            pass
        elif lb >= buffered_lb and ub <= buffered_ub:
            log.debug('Extracting from buffer')
            out[:] = self._buffer.get_range_samples(lb, ub)
            samples = 0
        elif lb >= buffered_lb and ub > buffered_ub:
            log.debug('Extracting from buffer and generating new data')
            b = self._buffer.get_range_samples(lb)
            s = b.shape[-1]
            out[:s] = b
            samples -= s
            #log.debug('Pulled %d samples out of buffer for %s', s, self.name)

        # Generate new samples
        if samples > 0:
            data = self.get_next_samples(samples)
            self._buffer.append_data(data)
            out[-samples:] = data

    def get_next_samples(self, samples):
        raise NotImplementedError


class EpochOutput(AnalogOutput):

    factory = Typed(Waveform)
    active = Bool(False)

    def _observe_factory(self, event):
        pass

    def get_next_samples(self, samples):
        if self.active:
            buffered_ub = self._buffer.get_samples_ub()

            # Pad with zero
            zero_padding = max(self._offset-buffered_ub, 0)
            zero_padding = min(zero_padding, samples)
            waveform_samples = samples - zero_padding

            waveforms = []
            if zero_padding:
                w = np.zeros(zero_padding, dtype=np.double)
                waveforms.append(w)
            if waveform_samples:
                w = self.factory.next(waveform_samples)
                waveforms.append(w)
            if self.factory.is_complete():
                self.deactivate(self._buffer.get_samples_ub())
            waveform = np.concatenate(waveforms, axis=-1)
        else:
            waveform = np.zeros(samples, dtype=np.double)
        return waveform

    def activate(self, offset):
        self.active = True
        self._offset = offset
        self._buffer.invalidate_samples(offset)

    def deactivate(self, offset):
        self.active = False
        self.factory = None
        self._buffer.invalidate_samples(offset)

    def _observe_token(self, event):
        if self.token is not None:
            self.token.configure_context_items(self.name, self.label, 'trial')

class QueuedEpochOutput(EpochOutput):

    queue = d_(Typed(AbstractSignalQueue))
    auto_decrement = d_(Bool(False))
    complete_cb = Typed(object)
    active = d_(Bool(True))

    def _observe_target(self, event):
        self._update_queue()

    def _observe_queue(self, event):
        self._update_queue()

    def _update_queue(self):
        if self.queue is not None and self.target is not None:
            self.queue.set_filter_delay(self.filter_delay)
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


class SelectorQueuedEpochOutput(QueuedEpochOutput):

    selector_name = d_(Unicode())


class ContinuousOutput(AnalogOutput):

    factory = Typed(Waveform)

    def get_next_samples(self, samples):
        return self.factory.next(samples)

    def _observe_token(self, event):
        if self.token is not None:
            self.token.configure_context_items(self.name, self.label, 'experiment')

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
