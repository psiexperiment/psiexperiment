import logging
log = logging.getLogger(__name__)

import numpy as np
from channel_plot import ChannelPlot
from traits.api import Float, Instance, Str, on_trait_change


class TimeseriesPlot(ChannelPlot):

    source = Instance('psi.data.hdf_store.data_source.DataTable')

    rect_height = Float(0.5)
    rect_center = Float(0.5)
    rising_event = Str()
    falling_event = Str()

    def _data_added(self, event):
        data_ub = event['value']['ub']
        data_event = event['value']['event']
        s_lb, s_ub = self.index_range.low, self.index_range.high
        if (s_lb <= data_ub < s_ub):
            if data_event in (self.rising_event, self.falling_event):
                self._invalidate_data()

    def _current_time_changed(self, event):
        self._invalidate_screen()
        self.request_redraw()

    def _source_changed(self, old, new):
        super(TimeseriesPlot, self)._source_changed(old, new)
        if old is not None:
            old.unobserve('current_time', self._current_time_changed)
        if new is not None:
            new.observe('current_time', self._current_time_changed)

    def _gather_points(self):
        if not self._data_cache_valid:
            epochs = self.source.get_epochs(self.rising_event,
                                            self.falling_event,
                                            self.index_range.low,
                                            self.index_range.high)
            self._cached_data = epochs
            self._data_cache_valid = True
            self._screen_cache_valid = False

        return self._cached_data

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            if len(self._cached_data) == 0:
                s_starts = []
                s_ends = []
            else:
                data = self._cached_data.copy()
                if np.isnan(data[-1, 1]):
                    data[-1, 1] = self.source.current_time
                s_starts = self.index_mapper.map_screen(data[:, 0])
                s_ends = self.index_mapper.map_screen(data[:, 1])
            self._cached_screen_index = s_starts, s_ends
            self._screen_cache_valid = True
            
        return self._cached_screen_index

    def _render(self, gc, points):
        starts, ends = points
        n = len(starts)
        if n == 0:
            return

        ttl_low = self.rect_center-self.rect_height*0.5
        ttl_high = self.rect_center+self.rect_height*0.5
        screen_low = self.value_mapper.map_screen(ttl_high)
        screen_high = self.value_mapper.map_screen(ttl_low)
        screen_height = screen_high-screen_low

        x = starts
        width = ends-starts
        y = np.ones(n) * screen_low
        height = np.ones(n) * screen_height

        with gc:
            gc.set_antialias(True)
            gc.clip_to_rect(self.x, self.y, self.width, self.height)

            # Set up appearance
            gc.set_stroke_color(self.line_color_)
            gc.set_fill_color(self.fill_color_)
            gc.set_line_width(self.line_width) 
            gc.set_line_dash(self.line_style_)
            gc.set_line_join(0) # Curved

            gc.begin_path()
            gc.rects(np.column_stack((x, y, width, height)))
            gc.draw_path()

            self._draw_default_axes(gc)

    def _index_mapper_updated(self):
        if self.source is not None:
            self._data_cache_valid = False
            self._invalidate_screen()
