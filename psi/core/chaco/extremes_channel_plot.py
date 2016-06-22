from __future__ import division

import numpy as np
from channel_plot import ChannelPlot
from traits.api import Any, Property, cached_property, Int
from traitsui.api import View, Item

import logging
log = logging.getLogger(__name__)

def decimate_rms(data, downsample):
    # If data is empty, return imediately
    if data.shape[-1] == 0:
        return [], []

    downsample *= 100

    # Determine the "fragment" size that we are unable to decimate.  A
    # downsampling factor of 5 means that we perform the operation in chunks of
    # 5 samples.  If we have only 13 samples of data, then we cannot decimate
    # the last 3 samples and will simply discard them. 
    last_dim = data.ndim
    offset = data.shape[-1] % downsample

    # Force a copy to be made, which speeds up min()/max().  Apparently min/max
    # make a copy of a reshaped array before performing the operation, so we
    # force it now so the copy only occurs once.
    if data.ndim == 2:
        shape = (len(data), -1, downsample)
    else:
        shape = (-1, downsample)
    data = data[..., :-offset].reshape(shape).copy()
    return np.abs(data).max(last_dim)

def decimate_simple(data, downsample):
    if data.shape[-1] == 0:
        return [], []
    # Determine the "fragment" size that we are unable to decimate.  A
    # downsampling factor of 5 means that we perform the operation in chunks of
    # 5 samples.  If we have only 13 samples of data, then we cannot decimate
    # the last 3 samples and will simply discard them. 
    return data[..., ::downsample]

def decimate_extremes(data, downsample):
    # If data is empty, return imediately
    #if data.shape[-1] == 0:
    #    return [], []
    if data.size == 0:
        return np.array([]), np.array([])

    # Determine the "fragment" size that we are unable to decimate.  A
    # downsampling factor of 5 means that we perform the operation in chunks of
    # 5 samples.  If we have only 13 samples of data, then we cannot decimate
    # the last 3 samples and will simply discard them. 
    last_dim = data.ndim
    offset = data.shape[-1] % downsample

    # Force a copy to be made, which speeds up min()/max().  Apparently min/max
    # make a copy of a reshaped array before performing the operation, so we
    # force it now so the copy only occurs once.
    if data.ndim == 2:
        shape = (len(data), -1, downsample)
    else:
        shape = (-1, downsample)
    data = data[..., :-offset].reshape(shape).copy()
    return data.min(last_dim), data.max(last_dim)

class ExtremesChannelPlot(ChannelPlot):
    
    _cached_min     = Any
    _cached_max     = Any

    # At what point should we switch from generating a decimated plot to a
    # regular line plot?
    dec_threshold = Int(6)
    draw_mode = Property(depends_on='dec_threshold, dec_factor')

    def _invalidate_data(self):
        self._cached_min = None
        self._cached_max = None
        super(ExtremesChannelPlot, self)._invalidate_data()

    def _data_changed(self):
        self._invalidate_data()

    def _dec_points_changed(self):
        # Flush the downsampled cache since it is no longer valid
        self._cached_min = None
        self._cached_max = None

    @cached_property
    def _get_draw_mode(self):
        return 'ptp' if self.dec_factor >= self.dec_threshold else 'normal'

    def _index_mapper_updated(self):
        super(ExtremesChannelPlot, self)._index_mapper_updated()
        self._cached_min = None
        self._cached_max = None

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            if self._cached_data.shape[-1] == 0:
                self._cached_screen_data = [], []
                self._cached_screen_index = []
            else:
                if self.draw_mode == 'normal':
                    self._compute_screen_points_normal()
                else:
                    self._compute_screen_points_decimated()
        return self._cached_screen_index, self._cached_screen_data

    def _compute_screen_points_normal(self):
        mapped = self._map_screen(self._cached_data)
        t = self.index_values[:mapped.shape[-1]]
        t_screen = self.index_mapper.map_screen(t)
        self._cached_screen_data = mapped 
        self._cached_screen_index = t_screen
        self._screen_cache_valid = True

    def _map_screen(self, data):
        return self.value_mapper.map_screen(data)

    def _compute_screen_points_decimated(self):
        # We cache our prior decimations 
        if self._cached_min is not None:
            n_cached = self._cached_min.shape[-1]*self.dec_factor
            to_decimate = self._cached_data[..., n_cached:]
            mins, maxes = decimate_extremes(to_decimate, self.dec_factor)
            self._cached_min = np.hstack((self._cached_min, mins))
            self._cached_max = np.hstack((self._cached_max, maxes))
        else:
            ptp = decimate_extremes(self._cached_data, self.dec_factor)
            self._cached_min = ptp[0]
            self._cached_max = ptp[1]

        # Now, map them to the screen
        samples = self._cached_min.shape[-1]
        s_val_min = self._map_screen(self._cached_min)
        s_val_max = self._map_screen(self._cached_max)
        self._cached_screen_data = s_val_min, s_val_max

        total_samples = self._cached_data.shape[-1]
        t = self.index_values[:total_samples:self.dec_factor][:samples]
        t_screen = self.index_mapper.map_screen(t)
        self._cached_screen_index = t_screen
        self._screen_cache_valid = True

    def _render(self, gc, points):
        if len(points[0]) == 0:
            return

        with gc:
            gc.clip_to_rect(self.x, self.y, self.width, self.height)
            gc.set_stroke_color(self.line_color_)
            gc.set_line_width(self.line_width) 

            gc.begin_path()
            if self.draw_mode == 'normal':
                idx, val = points
                gc.lines(np.c_[idx, val])
            else:
                idx, (mins, maxes) = points
                starts = np.column_stack((idx, mins))
                ends = np.column_stack((idx, maxes))
                gc.line_set(starts, ends)

            gc.stroke_path()
            self._draw_default_axes(gc)

    traits_view = View(
            Item('dec_points', label='Samples per pixel'),
            Item('dec_threshold', label='Decimation threshold'),
            Item('draw_mode', style='readonly'),
            Item('line_color'),
            Item('line_width'),
            )
