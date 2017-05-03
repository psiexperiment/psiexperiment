

from .base_channel_plot import BaseChannelPlot
import numpy as np
from traits.api import Bool, Property, cached_property, Int

import logging
log = logging.getLogger(__name__)

class ChannelPlot(BaseChannelPlot):
    '''
    Designed for efficiently handling time series data stored in a channel.
    Each time a Channel.updated event is fired, the new data is obtained and
    plotted.
    '''
    _data_cache_valid       = Bool(False)
    _screen_cache_valid     = Bool(False)

    def __init__(self, **kwargs):
        super(ChannelPlot, self).__init__(**kwargs)
        self._index_mapper_changed(None, self.index_mapper)

    def _invalidate_data(self):
        self._data_cache_valid = False
        self.deferred_redraw()

    def _invalidate_screen(self):
        self._screen_cache_valid = False
        self.deferred_redraw()

    def _index_mapper_updated(self):
        '''
        Compute array of index values (i.e. the time of each sample that could
        be displayed in the visible range)
        '''
        if self.source is not None:
            fs = self.source.fs
            # Channels contain continuous data starting at t0.  We do not want
            # to compute time values less than t0.
            if self.index_range.low > self.source.t0:
                low = int(self.index_range.low*fs)
            else:
                low = int(self.source.t0*fs)
            high = int(self.index_range.high*fs)
            self.index_values = np.arange(low, high)/fs
            self._invalidate_data()

    def _index_mapper_changed(self, old, new):
        if old is not None:
            old.on_trait_change(self._index_mapper_updated, "updated",
                    remove=True)
        if new is not None:
            new.on_trait_change(self._index_mapper_updated, "updated")

    def _draw_plot(self, gc, view_bounds=None, mode="normal"):
        self._get_data_points()
        self._get_screen_points()
        self._render(gc)

    def _get_data_points(self):
        if not self._data_cache_valid:
            range = self.index_mapper.range
            self._cached_data = self.source.get_range(range.low, range.high)
            self._data_cache_valid = True
            self._screen_cache_valid = False

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            # Obtain cached data and map to screen
            val_pts = self._cached_data
            s_val_pts = self.value_mapper.map_screen(val_pts)
            self._cached_screen_data = s_val_pts

            # Obtain cached data bounds and create index points
            n = val_pts.shape[-1]
            t_screen = self.index_mapper.map_screen(self.index_values[:n])
            self._cached_screen_index = t_screen

            # Screen cache is valid
            self._screen_cache_valid = True

    def _render(self, gc):
        if len(self._cached_screen_index) == 0:
            return

        with gc:
            gc.set_antialias(True)
            gc.clip_to_rect(self.x, self.y, self.width, self.height)
            gc.set_stroke_color(self.line_color_)
            gc.set_line_width(self.line_width)
            gc.begin_path()
            gc.lines(np.c_[self._cached_screen_index,
                           self._cached_screen_data])
            gc.stroke_path()
            self._draw_default_axes(gc)

    def _data_added(self, event):
        # We need to be smart about the data added event.  If we're not tracking
        # the index range, then the data that has changed *may* be off-screen.
        # In which case, we're doing a *lot* of work to redraw the exact same
        # picture.
        data_ub = event['value']['ub']
        s_lb, s_ub = self.index_range.low, self.index_range.high
        if s_lb <= data_ub < s_ub:
            self._invalidate_data()

    def _data_changed(self):
        self._invalidate_data()
