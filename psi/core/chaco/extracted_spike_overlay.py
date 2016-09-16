import numpy as np
from enable.api import ColorTrait
from chaco.api import AbstractOverlay
from traits.api import Instance, Array, Int
from enable import markers

from cns import get_config
colors = get_config('PAIRED_COLORS_RGB_NORM')

# The keys are values defined by UMS2000.  As long as the users has not modified 
# the default labels available, they will map to the label in the comment next
# to each entry
cluster_type_marker = {
    1:  (3,  markers.CIRCLE_MARKER),            # in process
    2:  (7,  markers.INVERTED_TRIANGLE_MARKER), # good unit
    3:  (7,  markers.TRIANGLE_MARKER),          # multi-unit
    4:  (1,  markers.DOT_MARKER),               # garbage
    5:  (3,  markers.DOT_MARKER),               # needs outlier removal
}

class ExtractedSpikeOverlay(AbstractOverlay):
    '''
    Supports overlaying the spike times on a multichannel view.  The component
    must be a subclass of MultiChannelPlot.

    clusters
        One entry for each timestamp indicating the cluster that event belongs
        to
    
    cluster_ids
        List of all cluster IDs

    cluster_types
        List of the type of each cluster listed in cluster_ids.  Controls how
        the cluster is plotted on-screen
    '''

    plot = Instance('enable.api.Component')
    
    timestamps = Array(dtype='float')   # Time in seconds of each event
    channels = Array(dtype='int')       # 0-based channel for each event
    clusters = Array(dtype='int')       # Cluster ID for each event
    cluster_ids = Array(dtype='int')    # List of clusters
    cluster_types = Array(dtype='int')  # Type of cluster

    marker_size = Int(5)
    line_width = Int(0)
    line_color = ColorTrait('white')
    marker_color = ColorTrait('red')

    def overlay(self, component, gc, view_bounds=None, mode="normal"):
        plot = self.plot
        if len(plot.channel_visible) != 0:
            with gc:
                gc.clip_to_rect(component.x, component.y, component.width,
                                component.height)
                gc.set_line_width(self.line_width)
                gc.set_stroke_color(self.line_color_)

                low, high = plot.index_range.low, plot.index_range.high
                ts_mask = (self.timestamps >= low) & (self.timestamps <= high)

                i = 0
                for c_id, c_type in zip(self.cluster_ids, self.cluster_types):
                    marker_size, marker_id = cluster_type_marker[c_type]
                    if c_type in (2, 3):
                        color = colors[i]
                        i += 1
                    else:
                        color = colors[-1]
                    gc.set_fill_color(color)
                    c_mask = self.clusters == c_id

                    for o, n in zip(plot.screen_offsets, plot.channel_visible):
                        ch_mask = self.channels == n
                        mask = ch_mask & c_mask & ts_mask
                        ts = self.timestamps[mask]

                        ts_offset = np.ones(len(ts))*o
                        ts_screen = plot.index_mapper.map_screen(ts)
                        points = np.column_stack((ts_screen, ts_offset))
                        gc.draw_marker_at_points(points, marker_size, marker_id)
