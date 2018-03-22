import logging
log = logging.getLogger(__name__)

from types import GeneratorType
from functools import partial

import numpy as np

from atom.api import (Unicode, Enum, Typed, Property, Float, Int, Bool, List)

from enaml.application import deferred_call
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Extension

from ..util import coroutine
from .queue import AbstractSignalQueue

from psi.core.enaml.api import PSIContribution
from psi.token.primitives import Waveform

import time


class Synchronized(PSIContribution):

    outputs = Property()

    def _get_outputs(self):
        return self.children


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

    def _get_calibration(self):
        return self.channel.calibration


class AnalogOutput(Output):

    buffer_size = Property()

    _buffer = Typed(np.ndarray)
    _offset = Int(0)

    def _get_buffer_size(self):
        return self.channel.buffer_size

    def _default__buffer(self):
        buffer_samples = int(self.buffer_size*self.fs)
        return np.zeros(buffer_samples, dtype=np.double)

    def get_samples(self, offset, samples, out=None):
        # TODO: eventually push this buffering down to the lowest layer (i.e.,
        # token generation)? I need to think about this because one caveat is
        # that we need to deal with stuff like QueuedEpochutput. This seems
        # like a reasonably sensible place to handle caching for now.
        log.trace('Requested %d samples at %d for %s', samples, offset,
                  self.name)
        if out is None:
            out = np.empty(samples, dtype=np.double)
        buffer_size = self._buffer.size

        if offset > self._offset:
            # This breaks an implicit software contract.
            raise SystemError('Mismatch between offsets')

        # Pull out buffered samples
        if offset < self._offset:
            lb = buffer_size+(offset-self._offset)
            ub = min(lb + samples, buffer_size)
            buffered_samples = ub-lb
            out[:buffered_samples] = self._buffer[lb:ub]
            log.debug('Pulled %d samples out of buffer for %s',
                      buffered_samples, self.name)
            samples -= buffered_samples

        if samples < 0:
            raise SystemError('Invalid request for samples')

        # Generate new samples
        if samples > 0:
            data = self.get_next_samples(samples)
            if samples > buffer_size:
                self._buffer[:] = data[-buffer_size:]
            else:
                self._buffer[:-samples] = self._buffer[samples:]
                self._buffer[-samples:] = data
            self._offset += samples
            out[-samples:] = data

        return out

    def get_next_samples(self, samples):
        raise NotImplementedError


class EpochOutput(AnalogOutput):

    factory = Typed(Waveform)
    active = Bool(False)
    duration = Float()

    def get_next_samples(self, samples):
        if self.active:
            waveform = self.factory.next(samples)
            if self.factory.is_complete():
                self.deactivate()
        else:
            waveform = np.zeros(samples, dtype=np.double)
        return waveform

    def activate(self, offset):
        self.active = True
        self._offset = offset
        self._buffer.fill(0)

    def deactivate(self):
        self.factory = None
        self.active = False


class QueuedEpochOutput(EpochOutput):

    selector_name = d_(Unicode())
    queue = d_(Typed(AbstractSignalQueue))
    auto_decrement = d_(Bool(False))
    complete_cb = Typed(object)

    def get_next_samples(self, samples):
        log.debug('Getting samples from queue')
        samples, empty = self.queue.pop_buffer(samples, self.auto_decrement)
        if empty and self.complete_cb is not None:
            deferred_call(self.complete_cb)
        return samples


class ContinuousOutput(AnalogOutput):

    factory = Typed(Waveform)

    def get_next_samples(self, samples):
        return self.factory.next(samples)


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
