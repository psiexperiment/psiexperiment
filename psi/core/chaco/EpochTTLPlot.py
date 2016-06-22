import numpy as np
from channel_plot import ChannelPlot
from traits.api import Float

class TTLPlot(ChannelPlot):

    rect_height = Float(0.5)
    rect_center = Float(0.5)

    def _get_screen_points(self):
        if not self._screen_cache_valid:
            # Obtain cached data and map to screen
            val_pts = self._cached_data
            idx_pts = self.index_values[:len(val_pts)]

            # Add a 0 to beginning of val_pts so we always ensure that a start
            # comes before an end.  Starts are where the TTL goes high, ends are
            # where the TTL goes low.
            val_pts = np.r_[0, val_pts]
            dv = np.diff(val_pts)
            starts = dv == 1
            ends = dv == -1

            # Ensure that we have an end for plotting purposes (if the TTL ends
            # in a high, then we just "stick" an end on and it will appear as
            # the leading edge of the TTL).
            if val_pts[-1] == 1:
                ends[-1] = True

            # Obtain cached data bounds and create index points
            t_starts = idx_pts[starts]
            t_ends = idx_pts[ends]
            s_t_starts = self.index_mapper.map_screen(t_starts)
            s_t_ends = self.index_mapper.map_screen(t_ends)
            self._cached_screen_index = s_t_starts, s_t_ends

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
