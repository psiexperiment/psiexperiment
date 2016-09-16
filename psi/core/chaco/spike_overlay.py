import numpy as np
from enable.api import MarkerTrait, ColorTrait
from chaco.api import AbstractOverlay
from traits.api import Instance, Any, Int
from traitsui.api import View, Item, VGroup

class SpikeOverlay(AbstractOverlay):

    plot = Instance('enable.api.Component')
    spikes = Any

    marker = MarkerTrait('inverted_triangle')
    marker_size = Int(5)
    line_width = Int(0)
    fill_color = ColorTrait('red')
    line_color = ColorTrait('white')

    def overlay(self, component, gc, view_bounds=None, mode="normal"):
        plot = self.plot
        if len(plot.channel_visible) != 0:
            with gc:
                gc.clip_to_rect(component.x, component.y, component.width,
                                component.height)
                gc.set_line_width(self.line_width)
                gc.set_fill_color(self.fill_color_)
                gc.set_stroke_color(self.line_color_)

                for o, n in zip(plot.screen_offsets, plot.channel_visible):
                    spikes = self.spikes[n]
                    ts = spikes.timestamps[:]/spikes.fs
                    ts_offset = np.ones(len(ts))*o
                    ts_screen = plot.index_mapper.map_screen(ts)
                    points = np.column_stack((ts_screen, ts_offset))
                    gc.draw_marker_at_points(points, self.marker_size,
                            self.marker_.kiva_marker)

    traits_view = View(
        VGroup(
            Item('marker'),
            Item('marker_size'),
            Item('line_width'),        
            Item('fill_color'),
            Item('line_color'),
            show_border=True,
            label='Spike Marker',
            ),
        )
