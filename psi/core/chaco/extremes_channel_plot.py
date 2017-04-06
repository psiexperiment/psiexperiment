

import numpy as np
from .channel_plot import ChannelPlot
from traits.api import Any, Property, cached_property, Int, Bool
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
    offset = int(data.shape[-1] % downsample)

    if data.ndim == 2:
        shape = (len(data), -1, downsample)
    else:
        shape = (-1, downsample)
    data = data[..., :-offset].reshape(shape)
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
    if data.size == 0:
        return np.array([]), np.array([])

    # Determine the "fragment" size that we are unable to decimate.  A
    # downsampling factor of 5 means that we perform the operation in chunks of
    # 5 samples.  If we have only 13 samples of data, then we cannot decimate
    # the last 3 samples and will simply discard them. 
    last_dim = data.ndim
    offset = int(data.shape[-1] % downsample)

    if data.ndim == 2:
        shape = (len(data), -1, downsample)
    else:
        shape = (-1, downsample)
    data = data[..., :-offset].reshape(shape)
    return data.min(last_dim), data.max(last_dim)


class ExtremesChannelPlot(ChannelPlot):
    
    _cached_min = Any
    _cached_max = Any
    _dec_cache_valid = Bool(False)

    # At what point should we switch from generating a decimated plot to a
    # regular line plot?
    dec_threshold = Int(6)

    # When decimating, how many samples should be extracted per pixel?
    dec_points = Int(2)
    dec_factor = Property(depends_on='index_mapper.updated, dec_points')

    draw_mode = Property(depends_on='dec_threshold, dec_factor')


    @cached_property
    def _get_dec_factor(self):
        '''
        Compute decimation factor based on the sampling frequency of the channel
        itself.
        '''
        screen_min, screen_max = self.index_mapper.screen_bounds
        screen_width = screen_max-screen_min # in pixels
        range = self.index_range
        data_width = (range.high-range.low)*self.source.fs
        return int(np.floor((data_width/screen_width)/self.dec_points))

    def _invalidate_dec_cache(self):
        self._dec_cache_valid = False

    def _data_changed(self):
        self._invalidate_dec_cache()
        self._invalidate_data()

    def _dec_points_changed(self):
        self._invalidate_dec_cache()

    @cached_property
    def _get_draw_mode(self):
        return 'ptp' if self.dec_factor >= self.dec_threshold else 'normal'

    def _index_mapper_updated(self):
        self._invalidate_dec_cache()
        super(ExtremesChannelPlot, self)._index_mapper_updated()

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            if self._cached_data.shape[-1] == 0:
                self._cached_screen_data = [], []
                self._cached_screen_index = []
            elif self.dec_factor >= self.dec_threshold:
                self._compute_screen_points_decimated()
            else:
                self._compute_screen_points_normal()
            self._screen_cache_valid = True

    def _compute_screen_points_decimated(self):
        # We cache our prior decimations 
        if self._dec_cache_valid:
            n_cached = int(self._cached_min.shape[-1]*self.dec_factor)
            to_decimate = self._cached_data[..., n_cached:]
            mins, maxes = decimate_extremes(to_decimate, self.dec_factor)
            self._cached_min = np.hstack((self._cached_min, mins))
            self._cached_max = np.hstack((self._cached_max, maxes))
        else:
            mins, maxes = decimate_extremes(self._cached_data, self.dec_factor)
            self._cached_min = mins
            self._cached_max = maxes

        # Now, map them to the screen
        samples = self._cached_min.shape[-1]
        s_val_min = self.value_mapper.map_screen(self._cached_min)
        s_val_max = self.value_mapper.map_screen(self._cached_max)
        self._cached_screen_data = s_val_min, s_val_max

        total_samples = self._cached_data.shape[-1]
        t = self.index_values[:total_samples:self.dec_factor][:samples]
        t_screen = self.index_mapper.map_screen(t)
        self._cached_screen_index = t_screen

    def _render(self, gc):
        if len(self._cached_screen_index) == 0:
            return

        with gc:
            gc.clip_to_rect(self.x, self.y, self.width, self.height)
            gc.set_stroke_color(self.line_color_)
            gc.set_line_width(self.line_width) 
            gc.begin_path()

            if self.dec_factor >= self.dec_threshold:
                try:
                    idx = self._cached_screen_index
                    mins, maxes = self._cached_screen_data
                    starts = np.column_stack((idx, mins))
                    ends = np.column_stack((idx, maxes))
                    gc.line_set(starts, ends)
                except:
                    log.info('{} idx, {} mins, {} maxes' \
                             .format(idx.shape, mins.shape, maxes.shape))
                    raise
            else:
                idx = self._cached_screen_index
                val = self._cached_screen_data
                gc.lines(np.c_[idx, val])

            gc.stroke_path()
            self._draw_default_axes(gc)
