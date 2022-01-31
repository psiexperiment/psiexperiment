import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import (Bool, Float, Int, List, Property, Tuple, Typed, Str)
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_

from psiaudio.calibration import BaseCalibration, FlatCalibration
from .output import QueuedEpochOutput, ContinuousOutput, EpochOutput
from ..core.enaml.api import PSIContribution


class Channel(PSIContribution):

    #: Globally-unique name of channel used for identification
    name = d_(Str()).tag(metadata=True)

    #: Code assigned by subclasses to identify channel type
    type_code = Str()

    #: Unique reference label used for tracking identity throughout
    #: psiexperiment
    reference = Str().tag(metadata=True)

    #: Label of channel used in GUI
    label = d_(Str()).tag(metadata=True)

    #: Is channel active during experiment?
    active = Property().tag(metadata=True)

    # SI unit (e.g., V)
    unit = d_(Str()).tag(metadata=True)

    # Number of samples to acquire before task ends. Typically will be set to
    # 0 to indicate continuous acquisition.
    samples = d_(Int(0)).tag(metadata=True)

    # Used to properly configure data storage.
    dtype = d_(Str()).tag(metadata=True)

    # Parent engine (automatically derived by Enaml hierarchy)
    engine = Property().tag(metadata=True)

    # Calibration of channel
    calibration = d_(Typed(BaseCalibration, factory=FlatCalibration.unity))
    calibration.tag(metadata=True)

    # Can the user modify the channel calibration?
    calibration_user_editable = d_(Bool(False)).tag(metadata=True)

    filter_delay = d_(Float(0).tag(metadata=True))

    def _default_name(self):
        raise NotImplementedError

    def _observe_name(self, event):
        self.reference = self._default_reference()

    def _default_reference(self):
        return f'{self.type_code}::{self.name}'

    def _default_calibration(self):
        return UnityCalibration()

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

    def sync_start(self, channel):
        '''
        Synchronize with channel so that sampling begins at the same time

        Parameters
        ----------
        channel : instance of Channel
            Channel to synchronize with.
        '''

        raise NotImplementedError

    def _get_active(self):
        raise NotImplementedError

    def __str__(self):
        return self.label


class HardwareMixin(Declarative):

    fs = d_(Float()).tag(metadata=True)


class SoftwareMixin(Declarative):
    pass


class InputMixin(Declarative):

    inputs = List().tag(metadata=True)

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

    def add_callback(self, cb):
        from .input import Callback
        callback = Callback(function=cb)
        self.add_input(callback)


class AnalogMixin(Declarative):

    # Expected input range (min/max)
    expected_range = d_(Tuple()).tag(metadata=True)


class DigitalMixin(Declarative):

    dtype = 'bool'


class CounterMixin(Declarative):

    def _get_active(self):
        return True


class OutputMixin(Declarative):

    outputs = List().tag(metadata=True)
    buffer_size = Property().tag(metadata=True)

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

    type_code = 'hw_ao'

    def get_samples(self, offset, samples, out=None):
        if out is None:
            out = np.empty(samples, dtype=np.double)
        n_outputs = len(self.outputs)
        waveforms = np.empty((n_outputs, samples))
        for output, waveform in zip(self.outputs, waveforms):
            output.get_samples(offset, samples, out=waveform)
        return np.sum(waveforms, axis=0, out=out)


class SoftwareAOChannel(AnalogMixin, OutputMixin, SoftwareMixin, Channel):

    type_code = 'sw_ao'


class HardwareAIChannel(AnalogMixin, InputMixin, HardwareMixin, Channel):

    type_code = 'hw_ai'

    #: Gain in dB of channel (e.g., due to a microphone preamp). The signal
    #: will be scaled down before further processing.
    gain = d_(Float()).tag(metadata=True)


class SoftwareAIChannel(AnalogMixin, InputMixin, SoftwareMixin, Channel):

    type_code = 'sw_ai'

    # Gain in dB of channel (e.g., due to a microphone preamp). The signal will
    # be scaled down before further processing.
    gain = d_(Float()).tag(metadata=True)


class HardwareDOChannel(DigitalMixin, OutputMixin, HardwareMixin, Channel):

    type_code = 'hw_do'


class SoftwareDOChannel(DigitalMixin, OutputMixin, SoftwareMixin, Channel):

    type_code = 'sw_do'


class HardwareDIChannel(DigitalMixin, InputMixin, HardwareMixin, Channel):

    type_code = 'hw_di'


class SoftwareDIChannel(DigitalMixin, InputMixin, SoftwareMixin, Channel):

    type_code = 'sw_di'
