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


class EpochOutput(AnalogOutput):

    method = Enum('merge', 'replace', 'multiply')

    # Track the total number of samples that have been generated.
    _generated_samples = Int()

    # Total number of samples that need to be generated for the epoch.
    _epoch_samples = Int()

    _timer = Instance(QTimer)
    _active = Bool(False)
    _callback = Callable()

    def initialize(self, plugin):
        '''
        Set up the generator in preparation for producing the signal. This
        allows the generator to cache some potentially expensive computations
        in advance rather than just before we actually want the signal played.
        '''
        kwargs = {
            'workbench': plugin.workbench,
            'fs': self.channel.fs,
            'calibration': self.channel.calibration
        }
        self._generator = self._token.initialize_generator(**kwargs)
        self._callback = partial(plugin.ao_callback, self.name)

    def start(self, plugin, start, delay):
        '''
        Actually start the generator. It must have been initialized first.
        '''
        if self._generator is None:
            log.warn('{} was not initialized'.format(self.name))
            self.initialize(plugin)

        token_duration = self._token.get_duration(plugin.workbench)
        self._epoch_samples = int(token_duration*self.channel.fs)
        self._offset = int((start+delay)*self.channel.fs)
        self._generated_samples = 0
        try:
            self.update()
            self.engine.register_ao_callback(self._callback, self.channel.name)
            log.debug('Registered callback to retrieve remaining samples')
        except GeneratorExit:
            log.debug('All samples successfully generated')

        plugin.invoke_actions('{}_start'.format(self.name), start+delay)
        end = start+delay+token_duration
        delay_ms = int((end-plugin.get_ts())*1e3)
        self._timer = QTimer()
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(lambda: self.clear(plugin, end, 0.2))
        self._timer.setSingleShot(True)
        self._timer.start()
        self._active = True

    def clear(self, plugin, end, delay):
        if not self._active:
            log.debug('Token is not active')
            return

        log.debug('Clearing {} epoch'.format(self.name))
        if self._timer is not None and self._timer.isActive():
            log.debug('Canceling timer')
            self._timer.stop()

        offset = int((end+delay)*self.channel.fs)
        samples = self._get_samples(offset, blocked=False)
        waveform = np.zeros(samples)
        self.engine.modify_hw_ao(waveform, offset, self.name)
        if self._generator is not None:
            self.engine.unregister_ao_callback(self._callback,
                                               self.channel.name)
        plugin.invoke_actions('{}_end'.format(self.name), end+delay)
        self._generator = None
        self._active = False

    def _get_samples(self, offset, blocked=True):
        buffer_offset = offset - self.engine.hw_ao_buffer_offset
        max_samples = self.engine.hw_ao_buffer_samples-buffer_offset
        if blocked:
            return min(self._block_samples, max_samples)
        else:
            return max_samples

    def update(self, plugin=None):
        log.debug('Updating epoch output {}'.format(self.name))
        samples = self._get_samples(self._offset)
        if samples == 0:
            return
        kwargs = {'samples': samples, 'offset': self._generated_samples}
        try:
            waveform = self._generator.send(kwargs)
            log.debug('Modifying HW waveform at {}'.format(self._offset))
            self.engine.modify_hw_ao(waveform, self._offset, self.name)
            self._generated_samples += len(waveform)
            self._offset += len(waveform)
            if self._generated_samples >= self._epoch_samples:
                raise StopIteration
        except StopIteration:
            self._generator = None
            raise StopIteration

    def configure(self, plugin):
        self._block_samples = int(self.channel.fs*self.block_size)
        kwargs = {
            'workbench': plugin.workbench,
            'fs': self.channel.fs,
            'calibration': self.channel.calibration
        }
        self._token.initialize_generator(**kwargs)


class ContinuousOutput(AnalogOutput):

    def start(self, plugin):
        log.debug('Configuring continuous output {}'.format(self.name))
        cb = partial(plugin.ao_callback, self.name)
        self.engine.register_ao_callback(cb, self.channel.name)
        kwargs = {
            'workbench': plugin.workbench,
            'fs': self.channel.fs,
            'calibration': self.channel.calibration
        }
        self._generator = self._token.initialize_generator(**kwargs)
        self._offset = 0

    def _get_samples(self):
        # TODO: probably should have a more sophisticated algorithm?
        samples = self.engine.get_space_available(self.channel.name, self._offset)
        return min(samples, self._block_samples)

    def configure(self, plugin):
        self._block_samples = int(self.channel.fs*self.block_size)
        self.start(plugin)

    def update(self, plugin=None):
        kwargs = {'offset': self._offset, 'samples': self._get_samples()}
        waveform = self._generator.send(kwargs)
        self.engine.append_hw_ao(waveform)
        self._offset += len(waveform)


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
