import logging
log = logging.getLogger(__name__)

from types import GeneratorType
from functools import partial

from atom.api import (Unicode, Enum, Typed, Property, Float, observe, Callable,
                      Int, Bool)
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Plugin, Extension

from .channel import Channel


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
        parent = self.parent
        while True:
            if isinstance(parent, Channel):
                return parent
            else:
                parent = parent.parent


class AnalogOutput(Output):

    visible = d_(Bool(True))
    _generator = Typed(GeneratorType)
    _offset = Int()

    def configure(self, plugin):
        pass



class EpochOutput(AnalogOutput):

    method = Enum('merge', 'replace', 'multiply')
    _waveform_offset = Int()

    def start(self, plugin, start_ts):
        kwargs = {'workbench': plugin.workbench, 'fs': self.channel.fs}
        self._generator = self._token.initialize_generator(**kwargs)
        self._offset = int(start_ts + 0.25 * self.channel.fs)
        self._waveform_offset = 0
        self.update()

        cb = partial(plugin.ao_callback, self.name)
        self.engine.register_ao_callback(cb, self.channel.name)

    def _get_samples(self):
        buffer_offset = self._offset - self.engine.hw_ao_buffer_offset
        return self.engine.hw_ao_buffer_samples-buffer_offset

    def update(self):
        log.debug('Updating epoch output {}'.format(self.name))
        kwargs = {
            'offset': self._waveform_offset, 
            'samples': self._get_samples()
        }
        waveform = self._generator.send(kwargs)
        log.debug('Modifying HW waveform at {}'.format(self._offset))
        self.engine.modify_hw_ao(waveform, self._offset, method=self.method)
        self._waveform_offset += len(waveform)
        self._offset += len(waveform)


class ContinuousOutput(AnalogOutput):

    def start(self, plugin):
        log.debug('Configuring continuous output {}'.format(self.name))
        cb = partial(plugin.ao_callback, self.name)
        self.engine.register_ao_callback(cb, self.channel.name)
        kwargs = {'workbench': plugin.workbench, 'fs': self.channel.fs}
        self._generator = self._token.initialize_generator(**kwargs)
        self._offset = 0

    def _get_samples(self):
        return self.engine.get_space_available(self.channel.name, self._offset)

    def configure(self, plugin):
        self.start(plugin)

    def update(self):
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
