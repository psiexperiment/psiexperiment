import numpy as np

from .channel_plot import ChannelPlot
from enable.api import black_color_trait, MarkerTrait
from traits.api import Instance, Float

class TimeseriesPlot(ChannelPlot):
    '''
    Designed for efficiently handling time series data (specifically,
    timestamps), stored in a channel.  Each time a Channel.updated event is
    fired, the new data is obtained and plotted.
    '''

    # Override the default type of Channel defined in BaseChannelPlot
    source              = Instance('cns.channel.Timeseries')

    marker              = MarkerTrait
    marker_size         = Float(4.0)
    marker_color        = black_color_trait
    marker_edge_color   = black_color_trait
    marker_edge_width   = Float(1.0)
    marker_height       = Float(0.5)

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

    def _data_added(self, timestamps):
        # Only fire an update if the changed data is within bounds
        if self.index_range.mask_data(np.array(timestamps)).any():
            self.invalidate_draw()
            self._data_cache_valid = False
            self.request_redraw()
