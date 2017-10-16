import logging
log = logging.getLogger(__name__)

from types import GeneratorType
from functools import partial

import numpy as np

from atom.api import (Unicode, Enum, Typed, Property, Float, Int, Bool)

from enaml.application import deferred_call
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Extension

from ..util import coroutine
from .queue import AbstractSignalQueue

from psi.core.enaml.api import PSIContribution

import time


class Output(PSIContribution):

    target_name = d_(Unicode())

    channel = Property()
    target = Property()
    engine = Property()

    # These two are defined as properties because it's theoretically possible
    # for the output to transform these (e.g., an output could upsample
    # children or "equalize" something before passing it along).
    fs = Property()
    calibration = Property()

    # TODO: clean this up. it's sort of hackish.
    token_name = d_(Unicode())
    token = Typed(Declarative)

    def _observe_parent(self, event):
        self.target_name = event['value'].name

    def _get_target(self):
        if isinstance(self.parent, Extension):
            return None
        return self.parent

    def _set_target(self, target):
        self.set_parent(target)

    def _get_engine(self):
        return self.channel.parent

    def _get_channel(self):
        from .channel import Channel
        parent = self.parent
        while True:
            if isinstance(parent, Channel):
                return parent
            else:
                parent = parent.parent

    def _get_fs(self):
        return self.channel.fs

    def _get_calibration(self):
        return self.channel.calibration


class AnalogOutput(Output):

    buffer_size = Property()

    _buffer = Typed(np.ndarray)
    _offset = Int(0)
    _generator = Typed(GeneratorType)

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
        log.debug('Requested {} samples at {} for {}' \
                  .format(samples, offset, self.name))
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
            samples -= buffered_samples

        if samples < 0:
            raise SystemError('Invalid request for samples')

        # Generate new samples
        if samples > 0:
            data = self.get_next_samples(samples)
            if samples > buffer_size:
                self._buffer[:] = data[-buffer_size:]
            else:
                self._buffer = np.roll(self._buffer, -samples)
                self._buffer[-samples:] = data
            self._offset += samples
            out[-samples:] = data

        return out

    def get_next_samples(self, samples):
        raise NotImplementedError


class EpochOutput(AnalogOutput):

    generator = Typed(object)
    active = Bool(False)
    duration = Float()

    def get_next_samples(self, samples):
        if self.active:
            waveform, complete = self.generator.send({'samples': samples})
            if complete:
                self.deactivate()
                samples -= waveform.size
                waveform = np.pad(waveform, (0, samples), 'constant')
        else:
            waveform = np.zeros(samples, dtype=np.double)
        return waveform

    def activate(self, offset):
        self._offset = offset
        self.active = True
        self._buffer.fill(0)

    def deactivate(self):
        self.generator = None
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

    generator = Typed(object)

    def get_next_samples(self, samples):
        return self.generator.send({'samples': samples})


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
