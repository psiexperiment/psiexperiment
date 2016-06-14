# TODO: Implement channel calibration. This is inherently tied to the engine
# though.
from atom.api import Unicode, Enum, Typed, Tuple, Property
from enaml.core.api import Declarative, d_


class Channel(Declarative):

    channel = d_(Unicode())

    # For software-timed channels, set sampling frequency to 0.
    fs = d_(Typed(object))
    start_trigger = d_(Unicode())

    engine = Property()

    def _get_engine(self):
        return self.parent

    def configure(self, plugin):
        pass


class AIChannel(Channel):

    inputs = Property()
    expected_range = d_(Tuple())

    def _get_inputs(self):
        return self.children

    def configure(self, plugin):
        for input in self.inputs:
            input.configure(plugin)


class AOChannel(Channel):

    outputs = Property()
    expected_range = d_(Tuple())

    def _get_outputs(self):
        return self.children

    def configure(self, plugin):
        for output in self.outputs:
            output.configure(plugin)


class DIChannel(Channel):

    inputs = Property()

    def _get_inputs(self):
        return self.children

    def configure(self, plugin):
        for input in self.inputs:
            input.configure(plugin)


class DOChannel(Channel):

    outputs = Property()

    def _get_outputs(self):
        return self.children

    def configure(self, plugin):
        for output in self.outputs:
            output.configure(plugin)
