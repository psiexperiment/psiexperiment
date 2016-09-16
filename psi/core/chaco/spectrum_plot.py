import numpy as np
from traits.api import Instance, Bool
from .base_channel_plot import BaseChannelPlot
from cns.sigtools import rfft

class SpectrumPlot(BaseChannelPlot):

    _data_cache_valid       = Bool(False)
    _screen_cache_valid     = Bool(False)

    spectrum_range = Instance('chaco.api.DataRange1D')

    def _index_mapper_updated(self):
        self._data_cache_valid = False
        self.invalidate_and_redraw()

    def _gather_points(self):
        if not self._data_cache_valid:
            range = self.spectrum_range
            data = self.source.get_range(range.low, range.high, channels=5)
            print data.shape
            self._cached_data = rfft(data, self.source.fs)
            self._data_cache_valid = True
            self._screen_cache_valid = False

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            # Obtain cached data and map to screen
            freq, power, phase = self._cached_data
            print 'freq', freq.shape
            self._cached_screen_freq = self.index_mapper.map_screen(freq)
            self._cached_screen_power = self.value_mapper.map_screen(power) 
            self._screen_cache_valid = True
        return self._cached_screen_freq, self._cached_screen_power

    def _draw_plot(self, gc, view_bounds=None, mode="normal"):
        self._gather_points()
        points = self._get_screen_points()
        self._render(gc, points)

    def _render(self, gc, points):
        idx, val = points
        print 'rendering'
        print points
        if len(idx) == 0:
            return
        gc.save_state()
        gc.set_antialias(True)
        gc.clip_to_rect(self.x, self.y, self.width, self.height)
        gc.set_stroke_color(self.line_color_)
        gc.set_line_width(self.line_width) 
        gc.begin_path()
        gc.lines(np.column_stack((idx, val)))
        gc.stroke_path()
        self._draw_default_axes(gc)
        gc.restore_state()
