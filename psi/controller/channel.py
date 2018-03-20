import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import Unicode, Enum, Typed, Tuple, Property, List
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_

from .calibration import Calibration
from .output import QueuedEpochOutput, ContinuousOutput, EpochOutput
from ..util import coroutine


class Channel(Declarative):

    name = d_(Unicode()).tag(metadata=True)
    label = d_(Unicode()).tag(metadata=True)
    unit = d_(Unicode()).tag(metadata=True)

    # Device-specific channel identifier.
    channel = d_(Unicode()).tag(metadata=True)

    # For software-timed channels, set sampling frequency to 0.
    fs = d_(Typed(object)).tag(metadata=True)

    # Can be blank for no start trigger (i.e., acquisition begins as soon as
    # task begins)
    start_trigger = d_(Unicode()).tag(metadata=True)

    # Used to properly configure data storage.
    dtype = d_(Unicode()).tag(metadata=True)
    engine = Property()
    calibration = d_(Typed(Calibration)).tag(metadata=True)
    active = Property()

    def _get_engine(self):
        return self.parent

    def _set_engine(self, engine):
        self.set_parent(engine)

    def _get_active(self):
        raise NotImplementedError

    def configure(self, plugin):
        pass


class InputChannel(Channel):

    inputs = List()

    def _get_active(self):
        return len(self.inputs) > 0

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

    def configure(self, plugin):
        for input in self.inputs:
            log.debug('Configuring input {}'.format(input.name))
            input.configure(plugin)


class OutputChannel(Channel):

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

    def _get_buffer_size(self):
        return self.engine.get_buffer_size(self.name)


class AIChannel(InputChannel):

    TERMINAL_MODES = 'pseudodifferential', 'differential', 'RSE', 'NRSE'
    expected_range = d_(Tuple()).tag(metadata=True)
    terminal_mode = d_(Enum(*TERMINAL_MODES)).tag(metadata=True)
    terminal_coupling = d_(Enum(None, 'AC', 'DC', 'ground')).tag(metadata=True)


class AOChannel(OutputChannel):

    TERMINAL_MODES = 'pseudodifferential', 'differential', 'RSE'
    expected_range = d_(Tuple()).tag(metadata=True)
    terminal_mode = d_(Enum(*TERMINAL_MODES)).tag(metadata=True)

    def get_samples(self, offset, samples, out=None):
        if out is None:
            out = np.empty(samples, dtype=np.double)
        n_outputs = len(self.outputs)
        waveforms = np.empty((n_outputs, samples))
        for output, waveform in zip(self.outputs, waveforms):
            output.get_samples(offset, samples, out=waveform)
        return np.sum(waveforms, axis=0, out=out)


class DIChannel(InputChannel):
    pass


class DOChannel(Channel):
    pass
