import numpy as np
from channel_plot import ChannelPlot
from traits.api import Float, Instance, Str, on_trait_change


class TimeseriesPlot(ChannelPlot):

    source = Instance('psi.data.hdf_store.data_source.DataTable')

    rect_height = Float(0.5)
    rect_center = Float(0.5)
    rising_event = Str()
    falling_event = Str()

    def _current_time_changed(self, event):
        self._invalidate_screen()
        self.request_redraw()

    def _source_changed(self, old, new):
        if old is not None:
            old.unobserve('added', self._data_changed)
            old.unobserve('current_time', self._current_time_changed)
        if new is not None:
            new.observe('added', self._data_changed)
            new.observe('current_time', self._current_time_changed)

    def _data_changed(self, result):
        ts, event = result['value'][0]
        if event in (self.rising_event, self.falling_event):
            s_lb, s_ub = self.index_range.low, self.index_range.high
            if (s_lb <= ts) and (ts < s_ub):
                self._invalidate_data()
                self.request_redraw()

    def _gather_points(self):
        if not self._data_cache_valid:
            range = self.index_mapper.range
            query = '(event == name) & (timestamp >= lb) & (timestamp < ub)'
            condvars = dict(name=self.rising_event, lb=range.low, ub=range.high)
            starts = self.source.query(query, condvars, field='timestamp')
            condvars['name'] = self.falling_event
            ends = self.source.query(query, condvars, field='timestamp')
            self._cached_data = [starts, ends]
            self._data_cache_valid = True
            self._screen_cache_valid = False
        return self._cached_data

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            # Obtain cached data and map to screen
            starts, ends = self._cached_data
            if len(starts) == 0 and len(ends) == 0:
                # Guard to avoid raising an error in the next statement
                pass
            elif (len(starts) == len(ends)) and (starts[0] > ends[0]):
                    starts = np.r_[self.index_mapper.range._low_value, starts]
                    ends = np.r_[ends, self.source.current_time]
            elif len(starts) > len(ends):
                ends = np.r_[ends, self.source.current_time]
            elif len(starts) < len(ends):
                starts = np.r_[self.index_mapper.range._low_value, starts]

            s_starts = self.index_mapper.map_screen(starts)
            s_ends = self.index_mapper.map_screen(ends)

            self._cached_screen_index = s_starts, s_ends
            self._screen_cache_valid = True
            
        return self._cached_screen_index

    def _render_icon(self, gc, x, y, width, height):
        gc.save_state()
        try:
            gc.set_stroke_color(self.line_color_)
            gc.set_fill_color(self.fill_color_)
            gc.set_line_width(self.line_width) 
            gc.set_line_dash(self.line_style_)
            gc.set_line_join(0) # Curved
            gc.draw_rect((x, y, width, height))
        finally:
            gc.restore_state()

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

