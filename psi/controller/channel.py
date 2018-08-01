import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import Unicode, Typed, Tuple, Property, List, Float, Int
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_

from .calibration import Calibration
from .output import QueuedEpochOutput, ContinuousOutput, EpochOutput
from ..util import coroutine


class Channel(Declarative):

    # Globally-unique name of channel used for identification
    name = d_(Unicode()).tag(metadata=True)

    # Lable of channel used in GUI
    label = d_(Unicode()).tag(metadata=True)

    # Is channel active during experiment?
    active = Property()

    # SI unit (e.g., V)
    unit = d_(Unicode()).tag(metadata=True)

    # Number of samples to acquire before task ends. Typically will be set to
    # 0 to indicate continuous acquisition.
    samples = d_(Int(0)).tag(metadata=True)

    # Used to properly configure data storage.
    dtype = d_(Unicode()).tag(metadata=True)

    # Parent engine (automatically derived by Enaml hierarchy)
    engine = Property()

    # Calibration of channel
    calibration = d_(Typed(Calibration)).tag(metadata=True)

    # Is channel active during experiment?
    active = Property()

    def __init__(self, *args, **kwargs):
        # This is a hack due to the fact that name is defined as a Declarative
        # member and each Mixin will overwrite whether or not the name is
        # tagged.
        super().__init__(*args, **kwargs)
        self.members()['name'].tag(metadata=True)

    def _get_engine(self):
        return self.parent

    def _set_engine(self, engine):
        self.set_parent(engine)

    def configure(self):
        pass

    def _get_active(self):
        raise NotImplementedError


class HardwareMixin(Declarative):

    fs = d_(Float()).tag(metadata=True)


class SoftwareMixin(Declarative):
    pass


class InputMixin(Declarative):

    inputs = List()

    def _get_active(self):
        active = [i.active for i in self.inputs]
        return any(active)

    def add_input(self, i):
        if i in self.inputs:
            return
        self.inputs.append(i)
        i.source = self

    def remove_input(self, i):
        if i not in self.inputs:
            return
        self.inputs.remove(i)
        i.source = None

    def configure(self):
        for input in self.inputs:
            log.debug('Configuring input {}'.format(input.name))
            input.configure()


class AnalogMixin(Declarative):

    # Expected input range (min/max)
    expected_range = d_(Tuple()).tag(metadata=True)


class DigitalMixin(Declarative):

    dtype = 'bool'


class CounterMixin(Declarative):

    def _get_active(self):
        return True


class OutputMixin(Declarative):

    outputs = List()
    buffer_size = Property()

    def _get_active(self):
        return len(self.outputs) > 0

    def add_output(self, o):
        if o in self.outputs:
            return
        self.outputs.append(o)
        o.target = self

    def remove_output(self, o):
        if o not in self.outputs:
            return
        self.outputs.remove(o)
        o.target = None

    def add_queued_epoch_output(self, queue, auto_decrement=True):
        # Subclasses of Enaml Declarative will automatically insert themselves
        # as children of the parent when initialized.
        o = QueuedEpochOutput(queue=queue, auto_decrement=auto_decrement)
        self.add_output(o)
        return o

    def _get_buffer_size(self):
        return self.engine.get_buffer_size(self.name)


class CounterChannel(CounterMixin, Channel):
    pass


class HardwareAOChannel(AnalogMixin, OutputMixin, HardwareMixin, Channel):

    def get_samples(self, offset, samples, out=None):
        if out is None:
            out = np.empty(samples, dtype=np.double)
        n_outputs = len(self.outputs)
        waveforms = np.empty((n_outputs, samples))
        for output, waveform in zip(self.outputs, waveforms):
            output.get_samples(offset, samples, out=waveform)
        return np.sum(waveforms, axis=0, out=out)


class SoftwareAOChannel(AnalogMixin, OutputMixin, SoftwareMixin, Channel):
    pass


class HardwareAIChannel(AnalogMixin, InputMixin, HardwareMixin, Channel):

    # Gain in dB of channel (e.g., due to a microphone preamp). The signal will
    # be scaled down before further processing.
    gain = d_(Float()).tag(metadata=True)


class SoftwareAIChannel(AnalogMixin, InputMixin, SoftwareMixin, Channel):

    # Gain in dB of channel (e.g., due to a microphone preamp). The signal will
    # be scaled down before further processing.
    gain = d_(Float()).tag(metadata=True)


class HardwareDOChannel(DigitalMixin, OutputMixin, HardwareMixin, Channel):
    pass


class SoftwareDOChannel(DigitalMixin, OutputMixin, SoftwareMixin, Channel):
    pass


class HardwareDIChannel(DigitalMixin, InputMixin, HardwareMixin, Channel):
    pass


class SoftwareDIChannel(DigitalMixin, InputMixin, SoftwareMixin, Channel):
    pass
