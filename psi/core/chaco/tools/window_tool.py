# Enthought library imports
from enable.api import cursor_style_trait, Line
from traits.api import Any, Bool, Enum, Instance, Int, List, Trait, \
        Tuple, HasTraits, Float, Property, Dict, Event
from enable.component import Component

# Chaco imports
from chaco.api import AbstractOverlay

XY = Tuple(Float, Float)

class Window(HasTraits):

    mode            = Enum('INCLUDE', 'EXCLUDE')
    mode_color      = Dict({'INCLUDE':  'black', 'EXCLUDE':  'red',})
    component       = Instance(Component)

    _line           = Instance(Line, args=dict(line_width=3, vertex_size=6))
    line            = Property(Instance(Line))

    # We need to store the points in data coordinates since screen coordinates
    # can change at anytime if the plot is resized.  Everytime the plot is
    # resized, the line is redrawn.  Each time the line is requested, we map the
    # data coordinates to the screen coordinates and update the line points.
    _points         = List(XY)
    points          = Property(depends_on='_points')
    screen_points   = Property
    distance        = Property(Float)
    constrain       = Enum('X', 'Y', None)

    def __init__(self, component=None, **kw):
        # Need to ensure component is initialized first so that
        # _set_screen_points can accurately map the points if needed
        if 'component' in kw:
            component = kw['component']
        self.component = component
        HasTraits.__init__(self, **kw)

    def _get_screen_points(self):
        return self.component.map_screen(self.points)

    def _set_screen_points(self, points):
        self.points = [self._map_data(point) for point in points]

    def _get_points(self):
        return self._points

    def get_hoop(self):
        (x1, y1), (x2, y2) = self.points
        return x1, (y1+y2)/2.0, abs(y1-y2)/2.0
    
    def set_hoop(self, value):
        # x-value, center-volt, half-height
        x, m, h = value
        print "split", x, m, h
        self.points = (x, m-h), (x, m+h)

    def _set_points(self, points):
        self._points = self._constrain_coords(points[0], points)

    def update_point(self, i, point):
        self._points = self._constrain_coords(point, self.points)
        self._points[i] = point

    def update_screen_point(self, i, point):
        self.update_point(i, self._map_data(point))

    def _get_line(self):
        self._line.points = list(self.component.map_screen(self.points))
        self._line.line_color = self.mode_color[self.mode]
        return self._line

    def _get_distance(self):
        return abs(self.screen_points[0][1]-self.screen_points[1][1])

    def _constrain_coords(self, point, points):
        if self.constrain == 'X':
            return [(point[0], p[1]) for p in points]
        elif self.constrain == 'Y':
            return [(p[0], point[1]) for p in points]
        else:
            return points

    def _map_data(self, point):
        """ Maps values from screen space into data space.
        """
        index_mapper = self.component.index_mapper
        value_mapper = self.component.value_mapper
        if self.component.orientation == 'h':
            ndx = index_mapper.map_data(point[0])
            val = value_mapper.map_data(point[1])
        else:
            ndx = index_mapper.map_data(point[1])
            val = value_mapper.map_data(point[0])
        return (ndx, val)

    def _map_screen(self, point):
        """ Maps values from data space into screen space.
        """
        index_mapper = self.component.index_mapper
        value_mapper = self.component.value_mapper

        if self.component.orientation == 'h':
            x = index_mapper.map_screen(point[0])
            y = value_mapper.map_screen(point[1])
        else:
            y = index_mapper.map_screen(point[0])
            x = value_mapper.map_screen(point[1])
        return (x, y)

    # Can modify these hittests so they are more appropriate for the type of
    # window we are analyzing.
    def _hittest(self, event, distance=4):
        """ Determines if the pointer is near a specified window. 
        """
        for i, point in enumerate(self.screen_points):
            if self._point_hittest(point, event, distance):
                return i
        return None # If no _point_hittest passes

    def _point_hittest(self, point, event, distance):
        e_x, e_y = event.x, event.y
        p_x, p_y = point
        return abs(p_x-e_x) + abs(p_y-e_y) <= distance

class WindowTool(AbstractOverlay):
    """ 
    The base class for tools that allow the user to draw a series of windows.
    
    Primarily used for spike isolation and sorting, but can conceivably be
    useful for other scenarios as well.
    """
    updated = Event

    # The component that this tool overlays
    component = Instance(Component)
    
    # The current windows
    _windows = List(Window)
    
    # The event states are:
    #   normal: 
    #     The user may have selected a window, and is moving the cursor around.
    #   selecting: 
    #     The user has clicked down but hasn't let go of the button yet,
    #     and can still drag the window around.
    #   dragging: 
    #     The user has clicked on an existing point and is dragging it
    #     around.  When the user releases the mouse button, the tool returns
    #     to the "normal" state
    event_state = Enum("normal", "selecting", "dragging")

    # The pixel distance from a vertex that is considered 'on' the vertex.
    proximity_distance = Int(4)
    max_windows = Int(4)
    coordinates = Property

    # The data (index, value) position of the mouse cursor; this is used by various
    # draw() routines.
    mouse_position = Trait(None, None, Tuple)

    # The index of the vertex being dragged, if any.
    _dragged = Trait(None, None, Tuple(Int, Int))
    
    # Is the point being dragged is a newly placed point? This informs the 
    # "dragging" state about what to do if the user presses Escape while 
    # dragging.
    _drag_new_point = Bool(False)
    
    # The previous event state that the tool was in. This is used for states
    # that can be canceled (e.g., by pressing the Escape key), so that the
    # tool can revert to the correct state.
    _prev_event_state = Any

    # The cursor shapes to use for various modes
    
    # Cursor shape for non-tool use.
    original_cursor = cursor_style_trait("arrow")
    # Cursor shape for drawing.
    normal_cursor = cursor_style_trait("arrow")
    # Cursor shape for deleting points.
    delete_cursor = cursor_style_trait("arrow")
    # Cursor shape for moving points.
    move_cursor = cursor_style_trait("sizing")

    # The tool is initially invisible, because there is nothing to draw.
    visible = Bool(False)

    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------
    def __init__(self, component=None, **kwtraits):
        if "component" in kwtraits:
            component = kwtraits["component"]
        super(WindowTool, self).__init__(**kwtraits)
        self.component = component
        self.reset()
    
    #------------------------------------------------------------------------
    # Drawing tool methods
    #------------------------------------------------------------------------
    def reset(self):
        """ Resets the tool, throwing away any points, and making the tool
        invisible.
        """
        self._windows = []
        self.event_state = "normal"
        self.visible = False
        self.request_redraw()

    #def _deactivate(self, component=None):
    #    """ Called by a PlotComponent when this is no longer the active tool.
    #    """
    #    self.reset()

    def normal_left_down(self, event):
        """ Handles the left mouse button being pressed while the tool is
        in the 'normal' state.
        
        For an existing point, if the user is pressing the Control key, the
        point is deleted. Otherwise, the user can drag the point.
        
        For a new point, the point is added, and the user can drag it.
        """
        # Determine if the user is dragging/deleting an existing point, or
        # creating a new one
        over = self._over_window(event, self._windows)
        if over is not None:
            if event.control_down:
                self._windows.pop(over[0]) # Delete the window
                self.updated = self
                self.request_redraw()
            else:
                self.event_state = "dragging"
                self._dragged = over
                self._drag_new_point = False
                self.dragging_mouse_move(event)
        elif len(self._windows) < self.max_windows:
            start_xy = event.x, event.y
            window = Window(screen_points=[start_xy, start_xy],
                            component=self.component)
            if event.shift_down:
                window.mode = 'EXCLUDE'
            self._windows.append(window)

            self._dragged = -1, 1 #e.g. last "window"
            self._drag_new_point = True
            self.visible = True
            self.event_state = "dragging"
            self.dragging_mouse_move(event)

    def normal_mouse_move(self, event):
        """ Handles the user moving the mouse in the 'normal' state.
        
        When the user moves the cursor over an existing point, if the Control 
        key is pressed, the cursor changes to the **delete_cursor**, indicating
        that the point can be deleted. Otherwise, the cursor changes to the
        **move_cursor**, indicating that the point can be moved.
        
        When the user moves the cursor over any other point, the cursor
        changes to (or stays) the **normal_cursor**.
        """
        # If the user moves over an existing point, change the cursor to be the
        # move_cursor; otherwise, set it to the normal cursor
        over = self._over_window(event, self._windows)
        if over is not None:
            if event.control_down:
                event.window.set_pointer(self.delete_cursor)
            else:
                event.window.set_pointer(self.move_cursor)
        else:
            event.handled = False
            if len(self._windows) < self.max_windows:
                event.window.set_pointer(self.normal_cursor)
            else:
                event.window.set_pointer(self.original_cursor)
        self.request_redraw()
    
    def normal_draw(self, gc):
        """ Draws the line.
        """
        for window in self._windows:
            window.line._draw(gc)
    
    def normal_key_pressed(self, event):
        """ Handles the user pressing a key in the 'normal' state.
        
        If the user presses the Enter key, the tool is reset.
        """
        if event.character == "Enter":
            self._finalize_selection()
            self.reset()

    def normal_mouse_leave(self, event):
        """ Handles the user moving the cursor away from the tool area.
        """
        event.window.set_pointer("arrow")
        
    #------------------------------------------------------------------------
    # "dragging" state
    #------------------------------------------------------------------------
    def dragging_mouse_move(self, event):
        """ Handles the user moving the mouse while in the 'dragging' state.
        
        The screen is updated to show the new mouse position as the end of the
        line segment being drawn.
        """
        window, point = self._dragged
        self._windows[window].update_screen_point(point, (event.x, event.y))
        self.request_redraw()

    def dragging_left_up(self, event):
        """ Handles the left mouse coming up in the 'dragging' state. 
        
        Switches to 'normal' state.
        """
        if self._windows[self._dragged[0]].distance < 4:
            self._cancel_drag()
        else:
            self.event_state = "normal"
            self._dragged = None
            self.updated = self
    
    def dragging_key_pressed(self, event):
        """ Handles a key being pressed in the 'dragging' state.
        
        If the key is "Esc", the drag operation is canceled.
        """
        if event.character == "Esc":
            self._cancel_drag()
    
    def dragging_mouse_leave(self, event):
        """ Handles the mouse leaving the tool area in the 'dragging' state.
        The drag is canceled and the cursor changes to an arrow.
        """
        self._cancel_drag()
        event.window.set_pointer("arrow")

    def _cancel_drag(self):
        """ Cancels a drag operation.
        """
        if self._dragged != None:
            if self._drag_new_point:
                # Only remove the point if it was a newly-placed point
                self._windows.pop(self._dragged[0])
            self._dragged = None
        self.mouse_position = None
        self.event_state = "normal"
        self.request_redraw()

    #------------------------------------------------------------------------
    # override AbstractOverlay methods
    #------------------------------------------------------------------------
    def overlay(self, component, gc, view_bounds, mode="normal"):
        """ Draws this component overlaid on another component.
        Implements AbstractOverlay.
        """
        with gc:
            c = component
            gc.clip_to_rect(c.x, c.y, c.width-1, c.height-1)
            for window in self._windows:
                window.line._draw(gc)
    
    def request_redraw(self):
        """ Requests that the component redraw itself. 
        Overrides Enable Component.
        """
        self.component.invalidate_draw()
        self.component.request_redraw()

    def _over_window(self, event, windows):
        """
        Return the index of a point in *points* that *event* is 'over'.
        Returns None if there is no such point.
        """
        for i, window in enumerate(windows):
            point = window._hittest(event)
            if point is not None:
                return i, point
        return None
    
    windows = Property(depends_on='updated')
    
    def _get_windows(self):
        return [w.get_hoop() for w in self._windows]
        
    def _set_windows(self, value):
        windows = []
        for v in value:
            window = Window(component=self.component)
            window.set_hoop(v)
            windows.append(window)
        self._windows = windows
        self.request_redraw()