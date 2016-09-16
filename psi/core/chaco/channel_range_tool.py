from traits.api import Float, on_trait_change, Bool, Instance, Enum
from enable.api import BaseTool, KeySpec

class ChannelRangeTool(BaseTool):

    # Can the user adjust the trigger delay by dragging with the left mouse
    # button?  This is not always desirable, so we allow disabling of the drag
    # feature.
    allow_drag  = Bool(True)

    # To disable zooming along one of the axes, set the corresponding factor to
    # 1.0.
    index_factor = Float(1.5)
    value_factor = Float(1.5)

    key_decr_by_span = Instance(KeySpec, args=("Left",))
    key_incr_by_span = Instance(KeySpec, args=("Right",))
    key_jump_to_start = Instance(KeySpec, args=("s",))
    key_reset_trig_delay = Instance(KeySpec, args=("0",))

    key_incr_zoom_value = Instance(KeySpec, args=("Up",))
    key_decr_zoom_value = Instance(KeySpec, args=("Down",))

    drag_mode = Enum('trig_delay', 'trigger')

    def normal_key_pressed(self, event):
        range = self.component.index_mapper.range
        if self.key_reset_trig_delay.match(event):
            range.trig_delay = 0
        elif self.key_jump_to_start.match(event):
            range.trigger = 0
        elif self.key_incr_by_span.match(event):
            span = range.span
            range.trigger += span
        elif self.key_decr_by_span.match(event):
            span = range.span
            range.trigger -= span
        elif self.key_incr_zoom_value.match(event):
            self.zoom_in_value(self.value_factor)
        elif self.key_decr_zoom_value.match(event):
            self.zoom_out_value(self.value_factor)

    def normal_mouse_enter(self, event):
        if self.component._window is not None:
            self.component._window._set_focus()

    def normal_mouse_wheel(self, event):
        if event.mouse_wheel != 0:
            if event.control_down:
                self.zoom_index(event)
            else:
                self.zoom_value(event)
            event.handled = True

    def normal_left_down(self, event):
        if self.allow_drag:
            range = self.component.index_mapper.range
            if event.control_down:
                self.drag_mode = 'trig_delay'
                self._start_delay = range.trig_delay
            else:
                self.drag_mode = 'trigger'
            self._start_data_x = event.x
            self._start_value = getattr(range, self.drag_mode)
            self.event_state = "panning"
            event.window.set_pointer("hand")
            event.window.set_mouse_owner(self, event.net_transform())
            event.handled = True

    def panning_mouse_move(self, event):
        delta_screen = event.x-self._start_data_x
        data_0 = self.component.index_mapper.map_data(0)
        data_d = self.component.index_mapper.map_data(delta_screen)
        delta_data = data_d-data_0

        range = self.component.index_mapper.range
        if self.drag_mode == 'trig_delay':
            range.trig_delay = self._start_value+delta_data
        else:
            range.trigger = self._start_value-delta_data

    def zoom_index(self, event):
        if event.mouse_wheel < 0:
            self.zoom_in_index(self.index_factor)
        else:
            self.zoom_out_index(self.index_factor)

    def zoom_out_index(self, factor):
        self.component.index_mapper.range.span /= factor

    def zoom_in_index(self, factor):
        self.component.index_mapper.range.span *= factor

    def zoom_value(self, event):
        if event.mouse_wheel > 0:
            self.zoom_in_value(self.value_factor)
        else:
            self.zoom_out_value(self.value_factor)

    def zoom_out_value(self, factor):
        self.component.value_mapper.range.low_setting *= factor
        self.component.value_mapper.range.high_setting *= factor

    def zoom_in_value(self, factor):
        self.component.value_mapper.range.low_setting /= factor
        self.component.value_mapper.range.high_setting /= factor

    def panning_left_up(self, event):
        self._end_pan(event)

    def panning_mouse_leave(self, event):
        self._end_pan(event)

    def _end_pan(self, event):
        self.event_state = 'normal'
        event.window.set_pointer("arrow")
        if event.window.mouse_owner == self:
            event.window.set_mouse_owner(None)
        event.handled = True

class MultiChannelRangeTool(ChannelRangeTool):

    value_span = Float(0.5e-3)
    value_factor = Float(1.1)

    def get_value_zoom_factor(self, event):
        return self.value_factor

    def normal_mouse_wheel(self, event):
        if event.mouse_wheel != 0:
            if event.control_down:
                if event.mouse_wheel < 0:
                    factor = self.get_value_zoom_factor(event)
                    self.zoom_in_index(factor)
                else:
                    factor = self.get_value_zoom_factor(event)
                    self.zoom_out_index(factor)
            else:
                if event.mouse_wheel < 0:
                    factor = self.get_value_zoom_factor(event)
                    self.zoom_out_value(factor)
                else:
                    factor = self.get_value_zoom_factor(event)
                    self.zoom_in_value(factor)

    def zoom_in_value(self, factor):
        self.value_span /= factor

    def zoom_out_value(self, factor):
        self.value_span *= factor

    @on_trait_change('value_span, component.channel_visible')
    def _update_span(self):
        span = self.value_span
        self.component.channel_offset = span/2.0
        self.component.channel_spacing = span
