from __future__ import division

import numpy as np
from channel_plot import ChannelPlot
from traits.api import Any, Property, cached_property, Int, Float
from traitsui.api import View, Item

import logging
log = logging.getLogger(__name__)


def decimate_rms(data, downsample):
    # If data is empty, return imediately
    if data.shape[-1] == 0:
        return np.array([])

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
    return np.mean((data**2), axis=last_dim)**0.5


class RMSChannelPlot(ChannelPlot):
    
    _cached_dec     = Any

    # At what point should we switch from generating a decimated plot to a
    # regular line plot?
    dec_threshold = Int(6)
    draw_mode = Property(depends_on='dec_threshold, dec_factor')

    dec_points = 1
    sensitivity = Float(5e-5)   # Volts/Pa
    input_gain = Float(57.0)    # In dB

    def _dec_points_changed(self):
        # Flush the downsampled cache since it is no longer valid
        self._cached_dec = None

    @cached_property
    def _get_draw_mode(self):
        return 'ptp' if self.dec_factor >= self.dec_threshold else 'normal'

    def _index_mapper_updated(self):
        super(RMSChannelPlot, self)._index_mapper_updated()
        self._cached_dec = None

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            if self._cached_data.shape[-1] == 0:
                self._cached_screen_data = []
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
        if self._cached_dec is not None:
            n_cached = self._cached_dec.shape[-1]*self.dec_factor
            to_decimate = self._cached_data[..., n_cached:]
            dec = decimate_rms(to_decimate, self.dec_factor)
            dec = 20*np.log10(dec/self.sensitivity) - self.input_gain
            self._cached_dec = np.hstack((self._cached_dec, dec))
        else:
            dec = decimate_rms(self._cached_data, self.dec_factor)
            dec = 20*np.log10(dec/self.sensitivity) - self.input_gain
            self._cached_dec = dec

        # Now, map them to the screen
        samples = self._cached_dec.shape[-1]
        self._cached_screen_data = self._map_screen(self._cached_dec)

        total_samples = self._cached_data.shape[-1]
        t = self.index_values[:total_samples:self.dec_factor][:samples]
        t_screen = self.index_mapper.map_screen(t)
        self._cached_screen_index = t_screen
        self._screen_cache_valid = True

    def _render(self, gc, points):
        if len(points[0]) == 0:
            return

        gc.save_state()
        gc.clip_to_rect(self.x, self.y, self.width, self.height)
        gc.set_stroke_color(self.line_color_)
        gc.set_fill_color(self.line_color_)
        gc.set_line_width(self.line_width) 

        gc.begin_path()
        idx, val = points
        log.debug('rendering idx with shape %r from %r', idx.shape, self.source)
        log.debug('rendering val with shape %r from %r', val.shape, self.source)
        gc.lines(np.c_[idx, val])
        gc.line_to(idx[-1], self.y)
        gc.line_to(idx[0], self.y)
        #gc.stroke_path()
        gc.fill_path()
        self._draw_default_axes(gc)
        gc.restore_state()

    traits_view = View(
            Item('dec_points', label='Samples per pixel'),
            Item('dec_threshold', label='Decimation threshold'),
            Item('draw_mode', style='readonly'),
            Item('line_color'),
            Item('line_width'),
            )
