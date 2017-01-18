import numpy as np

from traits.api import Float, Bool, Int

from neurogen.calibration.util import psd, psd_freq 
from .base_channel_plot import BaseChannelPlot


class FFTChannelPlot(BaseChannelPlot):

    time_span = Float(0.5)
    reference = Float(1)
    _current_slice = Int(-1)
    _data_cache_valid = Bool(False)
    _screen_cache_valid = Bool(False)

    def _invalidate_data(self):
        self._data_cache_valid = False
        self.deferred_redraw()

    def _data_changed(self, event):
        self._invalidate_data()

    def _data_added(self, event):
        ub = event['value']['ub']
        s = int(np.floor(ub/self.time_span)-1 )
        if  s > self._current_slice:
            self._current_slice = s
            self._invalidate_data()

    def _gather_points(self):
        if not self._data_cache_valid:
            start_slice = self._current_slice
            end_slice = start_slice + 1
            lb, ub = start_slice*self.time_span, end_slice*self.time_span
            data = self.source.get_range(lb, ub)
            if len(data) == 0:
                self._cached_data = np.array([]), np.array([])
            else:
                fs = self.source.fs
                freq = psd_freq(data, fs)
                power = 20*np.log10(psd(data, fs)/self.reference)
                self._cached_data = freq, power
                self._data_cache_valid = True
                self._screen_cache_valid = False

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            # Obtain cached data and map to screen
            idx_pts, val_pts = self._cached_data
            s_idx_pts = self.index_mapper.map_screen(idx_pts)
            s_val_pts = self.value_mapper.map_screen(val_pts)
            self._cached_screen_index = s_idx_pts
            self._cached_screen_data = s_val_pts
            # Screen cache is valid
            self._screen_cache_valid = True

        return self._cached_screen_index, self._cached_screen_data

    def _draw_plot(self, gc, view_bounds=None, mode="normal"):
        self._gather_points()
        points = self._get_screen_points()
        self._render(gc, points)

    def _render(self, gc, points):
        if len(points[0]) == 0:
            return
        with gc:
            gc.set_antialias(True)
            gc.clip_to_rect(self.x, self.y, self.width, self.height)
            gc.set_stroke_color(self.line_color_)
            gc.set_line_width(self.line_width)
            gc.begin_path()
            gc.lines(np.c_[points])
            gc.stroke_path()
            self._draw_default_axes(gc)
