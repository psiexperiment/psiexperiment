import logging
log = logging.getLogger(__name__)

from types import GeneratorType
from functools import partial

import numpy as np

from atom.api import (Unicode, Enum, Typed, Property, Float, observe, Callable,
                      Int, Bool, Instance, Callable)
from enaml.application import timed_call
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Plugin, Extension
from enaml.qt.QtCore import QTimer


class Output(Declarative):

    label = d_(Unicode())
    name = d_(Unicode())
    visible = d_(Bool(False))

    # TODO: Allow the user to select which channel the output goes through from
    # the GUI?
    target_name = d_(Unicode())

    channel = Property()
    target = Property()
    engine = Property()

    # TODO: clean this up. it's sort of hackish.
    _token_name = Unicode()
    _token = Typed(Declarative)

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


class AnalogOutput(Output):

    # This determines how many samples will be requested from the generator on
    # each iteration. While we can request an arbitrary number of samples, this
    # helps some caching mechanisms better cache the results for future calls.
    block_size = d_(Float(2))
    visible = d_(Bool(True))

    _generator = Typed(GeneratorType)
    _offset = Int()
    _block_samples = Int()

    def configure(self, plugin):
        pass

    def initialize_factory(self, context):
        context = context.copy()
        context['fs'] = self.channel.fs
        context['calibration'] = self.channel.calibration
        return self._token.initialize_factory(context)

    def initialize_generator(self, context):
        factory = self.initialize_factory(context)
        generator = factory()
        generator.next()
        return generator


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
        self.next()

    def next(self):
        if not self.active:
            raise StopIteration
        samples = \
            self.engine.get_buffered_samples(self.channel.name, self.offset)
        samples = min(self.output._block_samples, samples)
        waveform, complete = self.generator.send({'samples': samples})
        self.engine.modify_hw_ao(waveform, self.offset, self.output.name)
        self.offset += len(waveform)
        if complete:
            raise StopIteration

    def clear(self, end, delay):
        offset = int((end+delay)*self.channel.fs)
        samples = self.engine.get_buffered_samples(self.channel.name, offset)
        waveform = np.zeros(samples)
        self.engine.modify_hw_ao(waveform, offset, self.output.name)
        self.active = False


class EpochOutput(AnalogOutput):

    method = Enum('merge', 'replace', 'multiply')
    _cb = Typed(object)
    _duration = Typed(object)

    def initialize(self, plugin, context):
        '''
        Set up the generator in preparation for producing the signal. This
        allows the generator to cache some potentially expensive computations
        in advance rather than just before we actually want the signal played.
        '''
        # Load the context from the plugin if not provided already.
        generator = self.initialize_generator(context)
        cb = EpochCallback(self, generator)
        self._cb = cb
        self._duration = self._token.get_duration(context)

    def start(self, plugin, start, delay):
        '''
        Actually start the generator. It must have been initialized first.
        '''
        if self._cb is None:
            m = '{} was not initialized'.format(self.name)
            raise SystemError(m)
        self._cb.start(start, delay)
        self.engine.register_ao_callback(self._cb.next, self.channel.name)
        return self._duration

    def clear(self, plugin, end, delay):
        self._cb.clear(end, delay)

    def configure(self, plugin):
        self._block_samples = int(self.channel.fs*self.block_size)


def continuous_callback(output, generator):
    offset = 0
    engine = output.engine
    channel = output.channel
    while True:
        yield
        samples = engine.get_space_available(channel.name, offset)
        waveform = generator.send({'samples': samples})
        engine.append_hw_ao(waveform)
        offset += len(waveform)


class ContinuousOutput(AnalogOutput):

    def configure(self, plugin):
        log.debug('Configuring continuous output {}'.format(self.name))
        context = plugin.context.get_values()
        generator = self.initialize_generator(context)
        cb = continuous_callback(self, generator)
        cb.next()
        self.engine.register_ao_callback(cb.next, self.channel.name)


class DigitalOutput(Output):

    def configure(self, plugin):
        pass


class Trigger(DigitalOutput):

    duration = d_(Float(0.1))

    def fire(self):
        self.engine.fire_sw_do(self.channel.name, duration=self.duration)


class Toggle(DigitalOutput):

    def _set_state(self, state):
        self.engine.set_sw_do(self.channel.name, state)

    def set_high(self):
        self._set_state(1)

    def set_low(self):
        self._set_state(0)
