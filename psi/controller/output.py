import logging
log = logging.getLogger(__name__)

from functools import partial

from atom.api import Unicode, Enum, Typed, Property, Float, observe
from enaml.core.api import Declarative, d_
from enaml.workbench.api import Plugin, Extension

from .channel import Channel


class Output(Declarative):

    label = d_(Unicode())
    name = d_(Unicode())

    # TODO: Allow the user to select which channel the output goes through from
    # the GUI?
    target_name = d_(Unicode())

    channel = Property()
    target = Property()
    engine = Property()

    # TODO: clean this up. it's sort of hackish.
    _token_name = Unicode()
    _plugin_id = Unicode()
    _plugin = Typed(Plugin)

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
    pass


class EpochOutput(AnalogOutput):

    def get_waveform(self, offset=0, samples=None):
        return self._plugin.get_waveform(offset, samples)

    def configure(self, plugin):
        log.debug('Configuring epoch output {}'.format(self.name))
        self._plugin.initialize(self.channel.fs)


class ContinuousOutput(AnalogOutput):

    def get_waveform(self, offset, samples):
        return self._plugin.get_waveform(offset, samples)

    def configure(self, plugin):
        log.debug('Configuring continuous output {}'.format(self.name))
        cb = partial(plugin.ao_callback, self.name)
        self.engine.register_ao_callback(cb, self.channel.name)
        self._plugin.initialize(self.channel.fs)

    def update(self):
        offset = self.engine.get_offset(self.channel.name)
        samples = self.engine.get_space_available(self.channel.name, offset)
        log.trace('Generating {} samples at {} for {}' \
                  .format(samples, offset, self.name))
        waveform = self.get_waveform(offset, samples)
        self.engine.append_hw_ao(waveform)


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
