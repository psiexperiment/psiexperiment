from enable.api import ColorTrait, black_color_trait, color_table
from traits.api import (Property, Int, cached_property,
        on_trait_change, List)
import numpy as np
from base_channel_plot import BaseChannelPlot

class SnippetChannelPlot(BaseChannelPlot):

    last_reset  = Int(0)
    history = Int(20)
    classifier = Int(0)

    index_data = Property(depends_on='source.fs, source.snippet_size')
    index_screen = Property(depends_on='index_data')
    value_data = Property(depends_on='source.added, source.updated, last_reset, history')
    value_screen = Property(depends_on='value_data')
    classifier_masks = Property(depends_on='source.added, source.updated')

    colors = List(ColorTrait, ['red', 'green', 'blue', 'orange', 'black'])

    def _index_mapper_updated(self):
        pass

    @cached_property
    def _get_index_data(self):
        return np.arange(self.source.snippet_size)/self.source.fs

    @cached_property
    def _get_index_screen(self):
        return self.index_mapper.map_screen(self.index_data)

    @cached_property
    def _get_value_data(self):
        return self.source[self.last_reset:][-self.history:]

    @cached_property
    def _get_value_screen(self):
        return self.value_mapper.map_screen(self.value_data)

    @cached_property
    def _get_classifier_masks(self):
        classifiers = self.source.classifiers[self.last_reset:][-self.history:]
        return [c==classifiers for c in np.unique(classifiers)]

    def _configure_gc(self, gc):
        gc.save_state()
        gc.set_antialias(True)
        gc.clip_to_rect(self.x, self.y, self.width, self.height)
        gc.set_stroke_color(self.line_color_)
        gc.set_line_width(self.line_width) 

    def _draw_plot(self, gc, view_bounds=None, mode="normal"):
        self._render(gc)

    def _render(self, gc):
        index = self.index_screen
        if len(self.value_screen) == 0:
            return

        self._configure_gc(gc)
        gc.begin_path()
        
        for i, mask in enumerate(self.classifier_masks):
            color = color_table[self.colors[i]]
            gc.set_stroke_color(color)
            for value in self.value_screen[mask]:
                gc.lines(np.column_stack((index, value)))
            gc.stroke_path()

        self._draw_default_axes(gc)
        gc.restore_state()

    def _data_changed(self, event_data):
        self.invalidate_and_redraw()

    def _data_added(self, event_data):
        self.invalidate_and_redraw()

