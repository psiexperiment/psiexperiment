import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import (
    Bool, Float, Int, List, observe, Property, Tuple, Typed, set_default, Str
)
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_

from psiaudio.calibration import BaseCalibration, FlatCalibration
from psiaudio import util
from .output import QueuedEpochOutput, ContinuousOutput, EpochOutput
from ..core.enaml.api import PSIContribution


class Channel(PSIContribution):

    #: Globally-unique name of channel used for identification
    name = d_(Str()).tag(metadata=True)

    #: Code assigned by subclasses to identify channel type
    type_code = Str()

    #: Unique reference label used for tracking identity throughout
    #: psiexperiment.
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

    # Parent engine (automatically derived by Enaml hierarchy)
    engine = Property().tag(metadata=True)

    # Calibration of channel
    calibration = d_(Typed(BaseCalibration, factory=FlatCalibration.unity)) \
        .tag(metadata=True)

    # Can the user modify the channel calibration?
    calibration_user_editable = d_(Bool(False)).tag(metadata=True)

    filter_delay = d_(Float(0).tag(metadata=True))

    # Number of channels in the stream. This is for multichannel input that is
    # best processed as a group (e.g., from the Biosemi).
    n_channels = d_(Int(1)).tag(metadata=True)

    # Labels for channels
    channel_labels = d_(List()).tag(metadata=True)

    def _observe_name(self, event):
        self.reference = self._default_reference()

    def _default_reference(self):
        return f'{self.type_code}::{self.name}'

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

    def __repr__(self):
        return f'<Channel:{self.reference}>'


class HardwareMixin(Declarative):

    fs = d_(Float()).tag(metadata=True)

    def get_samples(self, offset, samples, out=None):
        '''
        Generate samples starting at offset
        '''
        if out is None:
            out = np.empty(samples, dtype=self.dtype)
        n_outputs = len(self.outputs)
        waveforms = np.empty((n_outputs, samples))
        for output, waveform in zip(self.outputs, waveforms):
            output.get_samples(offset, samples, out=waveform)
        return np.sum(waveforms, axis=0, out=out)


class SoftwareMixin(Declarative):

    fs = d_(Float(0)).tag(metadata=True, writable=False)


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

    # Used to properly configure data storage.
    dtype = d_(Str('double')).tag(metadata=True)

    #: Expected range of signal (min/max). Some backends (e.g., niDAQmx) can
    #: set hardware-based gains on certain channels (e.g., analog inputs and
    #: outputs of a PXI-4661) to optimize the SNR for the expected range.
    expected_range = d_(Tuple()).tag(metadata=True)

    #: Maximum allowable range of signal. This sets a hard upper bound on the
    #: expected range.
    max_range = d_(Tuple(default=(-np.inf, np.inf))).tag(metadata=True)

    @observe('expected_range', 'max_range')
    def _check_range(self, event):
        if not self.expected_range:
            return
        e_lb, e_ub = self.expected_range
        m_lb, m_ub = self.max_range
        valid = (m_lb <= e_lb < m_ub) and (m_lb < e_ub <= m_ub)
        if not valid:
            rel_db = util.db(self.expected_range[-1], self.max_range[-1])
            m = f'Expected range of {self.expected_range} ' \
                f'exceeds max range of {self.max_range} for {self}. ' \
                f'Try reducing your stimulus level by at least {rel_db:.1f} dB.'
            raise ValueError(m)


class DigitalMixin(Declarative):

    # Used to properly configure data storage.
    dtype = d_(Str('bool')).tag(metadata=True)


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

    def add_queued_epoch_output(self, queue, auto_decrement=True, name=None):
        # Subclasses of Enaml Declarative will automatically insert themselves
        # as children of the parent when initialized.
        if name is None:
            name = 'default'
        o = QueuedEpochOutput(queue=queue, auto_decrement=auto_decrement,
                              name=name)
        self.add_output(o)
        return o

    def _get_buffer_size(self):
        return self.engine.get_buffer_size(self.name)


class CounterChannel(CounterMixin, Channel):
    pass


class HardwareAOChannel(AnalogMixin, OutputMixin, HardwareMixin, Channel):

    type_code = set_default('hw_ao')


class SoftwareAOChannel(AnalogMixin, OutputMixin, SoftwareMixin, Channel):

    type_code = set_default('sw_ao')


class HardwareAIChannel(AnalogMixin, InputMixin, HardwareMixin, Channel):

    type_code = set_default('hw_ai')

    #: Gain in dB of channel (e.g., due to a microphone preamp). The signal
    #: will be scaled down before further processing.
    gain = d_(Float()).tag(metadata=True)


class SoftwareAIChannel(AnalogMixin, InputMixin, SoftwareMixin, Channel):

    type_code = set_default('sw_ai')

    # Gain in dB of channel (e.g., due to a microphone preamp). The signal will
    # be scaled down before further processing.
    gain = d_(Float()).tag(metadata=True)


class HardwareDOChannel(DigitalMixin, OutputMixin, HardwareMixin, Channel):

    type_code = set_default('hw_do')


class SoftwareDOChannel(DigitalMixin, OutputMixin, SoftwareMixin, Channel):

    type_code = set_default('sw_do')


class HardwareDIChannel(DigitalMixin, InputMixin, HardwareMixin, Channel):

    type_code = set_default('hw_di')


class SoftwareDIChannel(DigitalMixin, InputMixin, SoftwareMixin, Channel):

    type_code = set_default('sw_di')
