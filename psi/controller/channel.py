import logging
log = logging.getLogger(__name__)

import numpy as np

from atom.api import Unicode, Enum, Typed, Tuple, Property
from enaml.core.api import Declarative, d_

from neurogen.calibration import Calibration

from psi import SimpleState


class Channel(SimpleState, Declarative):

    label = d_(Unicode())

    # Device-specific channel identifier.
    channel = d_(Unicode())

    # For software-timed channels, set sampling frequency to 0.
    fs = d_(Typed(object))

    # Can be blank for no start trigger (i.e., acquisition begins as soon as
    # task begins)
    start_trigger = d_(Unicode())

    # Used to properly configure data storage.
    dtype = d_(Typed(np.dtype))

    engine = Property().tag(transient=True)

    def _get_engine(self):
        return self.parent

    def _set_engine(self, engine):
        self.set_parent(engine)

    def configure(self, plugin):
        pass


class AIChannel(Channel):

    inputs = Property().tag(transient=True)
    expected_range = d_(Tuple())
    calibration = Typed(Calibration)
    terminal_mode = d_(Enum('pseudodifferential', 'differential', 'RSE',
                            'NRSE'))
    terminal_coupling = d_(Enum(None, 'AC', 'DC', 'ground'))

    def _get_inputs(self):
        return self.children

    def configure(self, plugin):
        for input in self.inputs:
            log.debug('Configuring input {}'.format(input.name))
            input.configure(plugin)


class AOChannel(Channel):

    outputs = Property().tag(transient=True)
    expected_range = d_(Tuple())
    calibration = Typed(Calibration)
    terminal_mode = d_(Enum('pseudodifferential', 'differential', 'RSE'))

    def _get_outputs(self):
        return self.children

    def configure(self, plugin):
        for output in self.outputs:
            log.debug('Configuring output {}'.format(output.name))
            output.configure(plugin)


class DIChannel(Channel):

    inputs = Property().tag(transient=True)

    def _get_inputs(self):
        return self.children

    def configure(self, plugin):
        for input in self.inputs:
            input.configure(plugin)


class DOChannel(Channel):

    outputs = Property().tag(transient=True)

    def _get_outputs(self):
        return self.children

    def configure(self, plugin):
        for output in self.outputs:
            output.configure(plugin)
