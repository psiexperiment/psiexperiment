import numpy as np

from .timeseries_plot import TimeseriesPlot
from traits.api import Instance, Str, Any
from enable.api import black_color_trait, MarkerTrait
from traits.api import Instance, Float

from .channel_plot import ChannelPlot

class TablesTimeseriesPlot(ChannelPlot):

    source = Any
    trait_name = Str
    changed_name = Str

    marker              = MarkerTrait
    marker_size         = Float(4.0)
    marker_color        = black_color_trait
    marker_edge_color   = black_color_trait
    marker_edge_width   = Float(1.0)
    marker_height       = Float(0.5)


    def _source_changed(self, old, new):
        if old is not None:
            old.on_trait_change(self._data_changed, self.changed_name, True)
        if new is not None:
            new.on_trait_change(self._data_changed, self.changed_name)

    def _changed_name_changed(self, old, new):
        if self.source is not None:
            if old is not None:
                self.source.on_trait_change(self._data_changed, old, True)
            if new is not None:
                self.source.on_trait_change(self._data_changed, new)

    def _data_changed(self, result):
        ts, event = result
        if event == self.event_name:
            s_lb, s_ub = self.index_range.low, self.index_range.high
            if (s_lb <= ts) and (ts < s_ub):
                self._invalidate_data()
                self.request_redraw()

    def _gather_points(self):
        if not self._data_cache_valid:
            range = self.index_mapper.range
            query = '(event == name) & (ts >= lb) & (ts < ub)'
            condvars = dict(name=self.event_name, lb=range.low, ub=range.high)
            obj = getattr(self.source, self.trait_name)
            data = obj.read_where(query, condvars, field='ts')
            self._cached_data = data
            self._data_cache_valid = True
            self._screen_cache_valid = False
        return self._cached_data

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            screen_index = self.index_mapper.map_screen(self._cached_data)
            screen_position = self.value_mapper.map_screen(self.marker_height)
            screen_value = np.ones(len(screen_index))*screen_position
            self._cached_screen_points = np.c_[screen_index, screen_value]
            self._screen_cache_valid = True
        return self._cached_screen_points

    def _render(self, gc, points):
        if len(points) == 0:
            return

        gc.save_state()
        gc.set_antialias(True)
        gc.clip_to_rect(self.x, self.y, self.width, self.height)

        gc.set_fill_color(self.marker_color_)
        gc.set_stroke_color(self.marker_edge_color_)
        gc.set_line_width(self.marker_edge_width)
        gc.set_line_join(0) # Curved

        gc.draw_marker_at_points(points, self.marker_size,
                self.marker_.kiva_marker)

        self._draw_default_axes(gc)
        gc.restore_state()
