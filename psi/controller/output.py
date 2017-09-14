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

    visible = d_(Bool(False))

    target_name = d_(Unicode())

    channel = Property()
    target = Property()
    engine = Property()
    fs = Property()

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


class AnalogOutput(Output):

    # This determines how many samples will be requested from the generator on
    # each iteration. While we can request an arbitrary number of samples, this
    # helps some caching mechanisms better cache the results for future calls.
    block_size = d_(Float(2))
    visible = d_(Bool(True))

    buffer_size = Property()

    _buffer = Typed(np.ndarray)
    _offset = Int(0)
    _generator = Typed(GeneratorType)
    _block_samples = Int()

    def _get_buffer_size(self):
        return self.channel.buffer_size

    def _default__buffer(self):
        buffer_samples = int(self.buffer_size*self.fs)
        return np.zeros(buffer_samples, dtype=np.double)

    def configure(self, plugin):
        pass

    def initialize_factory(self, context):
        context = context.copy()
        context['fs'] = self.channel.fs
        context['calibration'] = self.channel.calibration
        return self.token.initialize_factory(context)

    def initialize_generator(self, context):
        factory = self.initialize_factory(context)
        generator = factory()
        next(generator)
        return generator

    def get_samples(self, offset, samples, out=None):
        # Draw from the buffer if needed.
        log.debug('Requested {} samples at {} for {}' \
                  .format(samples, offset, self.name))
        if out is None:
            out = np.empty(samples, dtype=np.double)
        buffer_size = self._buffer.size

        if self._offset < offset:
            # This breaks an implicit software contract.
            raise SystemError('Mismatch between offsets')

        # Pull out buffered samples
        if offset < self._offset:
            lb = buffer_size-self._offset-offset
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


class EpochCallback(object):

    def __init__(self, output, generator):
        self.output = output
        self.engine = output.engine
        self.channel = output.channel
        self.active = False
        self.generator = generator

    def start(self, start, delay):
        self.offset = int((start+delay)*self.channel.fs)
        self.active = True
        next(self)

    def send(self, event):
        if not self.active:
            raise StopIteration
        with self.engine.lock:
            samples = self.engine.get_buffered_samples(self.channel.name,
                                                       self.offset)
            samples = min(self.output._block_samples, samples)
            if samples == 0:
                return
            waveform, complete = self.generator.send({'samples': samples})
            self.engine.modify_hw_ao(waveform, self.offset, self.output.name)
            self.offset += len(waveform)
            if complete:
                raise StopIteration
 
    def clear(self, end, delay):
        with self.engine.lock:
            offset = int((end+delay)*self.channel.fs)
            samples = self.engine.get_buffered_samples(self.channel.name,
                                                       offset)
            waveform = np.zeros(samples)
            self.engine.modify_hw_ao(waveform, offset, self.output.name)
            self.active = False


class EpochOutput(AnalogOutput):

    method = d_(Enum('merge', 'replace', 'multiply'))

    _cb = Typed(object)
    _duration = Typed(object)

    def setup(self, context):
        '''
        Set up the generator in preparation for producing the signal. This
        allows the generator to cache some potentially expensive computations
        in advance rather than just before we actually want the signal played.
        '''
        raise NotImplementedError
        generator = self.initialize_generator(context)
        cb = EpochCallback(self, generator)
        self._cb = cb
        self._duration = self.token.get_duration(context)

    def start(self, start, delay):
        '''
        Actually start the generator. It must have been initialized first.
        '''
        if self._cb is None:
            m = '{} was not initialized'.format(self.name)
            raise SystemError(m)
        self._cb.start(start, delay)
        self.engine.register_ao_callback(self._cb.send, self.channel.name)
        return self._duration

    def clear(self, end, delay):
        self._cb.clear(end, delay)

    def configure(self, plugin):
        self._block_samples = int(self.channel.fs*self.block_size)


class QueuedEpochOutput(EpochOutput):

    selector_name = d_(Unicode())
    queue = d_(Typed(AbstractSignalQueue))
    auto_decrement = d_(Bool(False))
    complete_cb = Typed(object)

    def setup(self, context, complete_cb=None):
        log.debug('Setting up queue for {}'.format(self.name))
        self.complete_cb = complete_cb
        for setting in context:
            averages = setting['{}_averages'.format(self.name)]
            iti_duration = setting['{}_iti_duration'.format(self.name)]
            token_duration = self.token.get_duration(setting)
            delay_duration = iti_duration-token_duration
            # Somewhat surprisingly it appears to be faster to use factories in
            # the queue rather than creating the waveforms for ABR tone pips,
            # even for very short signal durations.
            factory = self.initialize_factory(setting)
            self.queue.append(factory, averages, delay_duration, token_duration, setting)

    def get_next_samples(self, samples):
        log.debug('Getting samples from queue')
        samples, empty = self.queue.pop_buffer(samples, self.auto_decrement)
        if empty and self.complete_cb is not None:
            deferred_call(self.complete_cb)
        return samples


class ContinuousOutput(AnalogOutput):

    _generator = Typed(object)

    def setup(self, context):
        log.debug('Configuring continuous output {}'.format(self.name))
        self._generator = self.initialize_generator(context)

    def get_next_samples(self, samples):
        return self._generator.send({'samples': samples})


class DigitalOutput(Output):

    def configure(self, plugin):
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
