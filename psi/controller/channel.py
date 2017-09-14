import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import Unicode, Enum, Typed, Tuple, Property
from enaml.application import deferred_call
from enaml.core.api import Declarative, d_

from .calibration import Calibration
from .output import ContinuousOutput, EpochOutput
from ..util import coroutine


class Channel(Declarative):

    label = d_(Unicode()).tag(metadata=True)

    # Device-specific channel identifier.
    channel = d_(Unicode()).tag(metadata=True)

    # For software-timed channels, set sampling frequency to 0.
    fs = d_(Typed(object)).tag(metadata=True)

    # Can be blank for no start trigger (i.e., acquisition begins as soon as
    # task begins)
    start_trigger = d_(Unicode()).tag(metadata=True)

    # Used to properly configure data storage.
    dtype = d_(Typed(np.dtype))

    engine = Property()

    calibration = d_(Typed(Calibration)).tag(metadata=True)

    def _get_engine(self):
        return self.parent

    def _set_engine(self, engine):
        self.set_parent(engine)

    def configure(self, plugin):
        pass


class InputChannel(Channel):

    inputs = Property()

    def _get_inputs(self):
        return self.children

    def configure(self, plugin):
        for input in self.inputs:
            log.debug('Configuring input {}'.format(input.name))
            input.configure(plugin)


class OutputChannel(Channel):

    outputs = Property()
    buffer_size = Property()

    def _get_outputs(self):
        return self.children

    def _get_buffer_size(self):
        return self.engine.get_buffer_size(self.name)

    def configure(self, plugin):
        for output in self.outputs:
            log.debug('Configuring output {}'.format(output.name))
            output.configure(plugin)


class AIChannel(InputChannel):

    TERMINAL_MODES = 'pseudodifferential', 'differential', 'RSE', 'NRSE'
    expected_range = d_(Tuple()).tag(metadata=True)
    terminal_mode = d_(Enum(*TERMINAL_MODES)).tag(metadata=True)
    terminal_coupling = d_(Enum(None, 'AC', 'DC', 'ground')).tag(metadata=True)


class AOChannel(OutputChannel):

    TERMINAL_MODES = 'pseudodifferential', 'differential', 'RSE'
    expected_range = d_(Tuple()).tag(metadata=True)
    terminal_mode = d_(Enum(*TERMINAL_MODES)).tag(metadata=True)

    def get_samples(self, offset, samples):
        n_outputs = len(self.outputs)
        waveforms = np.empty((n_outputs, samples))
        for i, output in enumerate(self.outputs):
            waveforms[i] = output.get_samples(offset, samples)
        return np.sum(waveforms, axis=0)


class DIChannel(InputChannel):
    pass


class DOChannel(Channel):
    pass
