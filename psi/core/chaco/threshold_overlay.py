from enable.api import ColorTrait
from chaco.api import AbstractOverlay
from traits.api import (Instance, Property, cached_property, Int,
        Float, List, Bool, on_trait_change)
from traitsui.api import View, VGroup, Item

class ThresholdOverlay(AbstractOverlay):

    plot = Instance('enable.api.Component')

    line_width = Int(1)
    line_color = ColorTrait('red')

    sort_thresholds = List(Float)
    sort_channels = List(Int)
    sort_signs = List(Bool)

    th_screen = Property(depends_on='sort_+, plot.channel_visible, plot.offsets')

    @on_trait_change('+', post_init=True)
    def _sort_settings_changed(self):
        self.plot.request_redraw()
    
    @cached_property
    def _get_th_screen(self):
        bounds = []
        for ch in self.sort_channels:
            if ch in self.plot.channel_visible:
                i = self.plot.channel_visible.index(ch)
                o = self.plot.offsets[i]
                th = self.sort_thresholds[ch]
                bounds.append(self.plot.value_mapper.map_screen(o+th))
                if not self.sort_signs[ch]:
                    bounds.append(self.plot.value_mapper.map_screen(o-th))
        return bounds
    
    def overlay(self, component, gc, view_bounds=None, mode="normal"):
        if len(self.plot.channel_visible) == 0:
            return
        with gc:
            gc.clip_to_rect(component.x, component.y, component.width,
                            component.height)
            gc.set_line_width(self.line_width)
            gc.set_stroke_color(self.line_color_)
            xlb = component.x
            xub = component.x+component.width
            for y in self.th_screen:
                gc.move_to(xlb, y)
                gc.line_to(xub, y)
            gc.stroke_path()                
                
    traits_view = View(
        VGroup(
            Item('line_color'),
            Item('line_width'),
            show_border=True,
            label='Threshold overlay',
            )            
        )
