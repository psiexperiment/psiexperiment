'''
Data-range state machines for the plotting subsystem.

These track the currently-visible x-range of a plot container based on the
data flowing through the attached sources. Pure Atom classes — no
pyqtgraph/Qt — so the windowing semantics can be tested without a GUI.
'''
from atom.api import Atom, Float, List, Tuple, Typed
from enaml.core.api import d_


class BaseDataRange(Atom):

    container = Typed(object)

    # Size of display window
    span = Float(1)

    # Delay before clearing window once data has scrolled off the window.
    delay = Float(0)

    # Current visible data range
    current_range = Tuple(Float(), Float())

    def add_source(self, source):
        raise NotImplementedError

    def _default_current_range(self):
        return 0, self.span

    def _observe_delay(self, event):
        if event['type'] == 'create':
            return
        self._update_range()

    def _observe_span(self, event):
        if event['type'] == 'create':
            return
        self._update_range()

    def _update_range(self):
        raise NotImplementedError

    def data_received(self, data):
        raise NotImplementedError


class EpochDataRange(BaseDataRange):

    max_duration = Float(0)

    def data_received(self, data):
        self.max_duration = max(data.duration, self.max_duration)

    def add_source(self, source):
        source.add_callback(self.data_received)

    def _observe_max_duration(self, event):
        self._update_range()

    def _update_range(self):
        self.current_range = 0, self.max_duration


class ChannelDataRange(BaseDataRange):

    # Automatically updated. Indicates last seen time based on the first data
    # source reporting to this range.
    current_time = Float(0)
    track_sources = d_(List())

    def _update_range(self):
        low_value = (self.current_time//self.span)*self.span - self.delay
        high_value = low_value+self.span
        if self.current_range != (low_value, high_value):
            self.current_range = low_value, high_value

    def data_received(self, data):
        # Invoked whenever a source recieves data (either Events or
        # PipelineData)
        self.current_time = max(self.current_time, data.t_end)
        lb = (self.current_time // self.span) * self.span - self.delay
        if self.current_range[0] != lb:
            self._update_range()

    def add_source(self, source):
        if self.track_sources:
            if source.name in self.track_sources:
                source.add_callback(self.data_received)
        else:
            source.add_callback(self.data_received)
