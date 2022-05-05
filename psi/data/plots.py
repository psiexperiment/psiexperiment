import logging

log = logging.getLogger(__name__)

import itertools
import importlib
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
import pyqtgraph as pg

from atom.api import (Str, Float, Tuple, Int, Typed, Property, Atom,
                      Bool, Enum, List, Dict, Callable, Value, observe)

from enaml.application import deferred_call, timed_call
from enaml.colors import parse_color
from enaml.core.api import Looper, Declarative, d_, d_func
from enaml.qt.QtGui import QColor

from psiaudio import util

from psi.util import SignalBuffer, ConfigurationException
from psi.core.enaml.api import load_manifests, PSIContribution
from psi.context.context_item import ContextMeta



################################################################################
# Utility functions
################################################################################
def get_x_fft(fs, duration):
    n_time = int(fs * duration)
    freq = np.fft.rfftfreq(n_time, fs**-1)
    return np.log10(freq)


def get_color_cycle(name):
    module_name, cmap_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cmap = getattr(module, cmap_name)
    return itertools.cycle(cmap.colors)


def make_color(color):
    if isinstance(color, tuple):
        return QColor(*color)
    elif isinstance(color, str):
        return QColor(color)
    else:
        raise ValueError('Unknown color %r', color)


################################################################################
# Style mixins
################################################################################
class ColorCycleMixin(Declarative):

    #: Define the pen color cycle. Can be a list of colors or a string
    #: indicating the color palette to use in palettable.
    pen_color_cycle = d_(Typed(object))
    _plot_colors = Typed(dict)

    def _default_pen_color_cycle(self):
        return ['k']

    def _make_plot_cycle(self):
        if isinstance(self.pen_color_cycle, str):
            cycle = get_color_cycle(self.pen_color_cycle)
        else:
            cycle = itertools.cycle(self.pen_color_cycle)
        return defaultdict(lambda: next(cycle))

    @d_func
    def get_pen_color(self, key):
        if self._plot_colors is None:
            self._plot_colors = self._make_plot_cycle()
        color = self._plot_colors[key]
        if not isinstance(color, str):
            return QColor(*color)
        else:
            return QColor(color)

    def _observe_pen_color_cycle(self, event):
        self._plot_colors = self._make_plot_cycle()
        self._reset_plots()

    def _reset_plots(self):
        raise NotImplementedError


################################################################################
# Supporting classes
################################################################################
class BaseDataRange(Atom):

    container = Typed(object)

    # Size of display window
    span = Float(1)

    # Delay before clearing window once data has "scrolled off" the window.
    delay = Float(0)

    # Current visible data range
    current_range = Tuple(Float(), Float())

    def add_source(self, source):
        cb = partial(self.source_added, source=source)
        source.add_callback(cb)

    def _default_current_range(self):
        return 0, self.span

    def _observe_delay(self, event):
        self._update_range()

    def _observe_span(self, event):
        self._update_range()

    def _update_range(self):
        raise NotImplementedError


class EpochDataRange(BaseDataRange):

    max_duration = Float()

    def source_added(self, data, source):
        n = [len(d['signal']) for d in data]
        max_duration = max(n) / source.fs
        self.max_duration = max(max_duration, self.max_duration)

    def _observe_max_duration(self, event):
        self._update_range()

    def _update_range(self):
        self.current_range = 0, self.max_duration


class ChannelDataRange(BaseDataRange):

    # Automatically updated. Indicates last "seen" time based on all data
    # sources reporting to this range.
    current_time = Float(0)

    current_samples = Typed(defaultdict, (int,))
    current_times = Typed(defaultdict, (float,))

    def _observe_current_time(self, event):
        self._update_range()

    def _update_range(self):
        low_value = (self.current_time//self.span)*self.span - self.delay
        high_value = low_value+self.span
        self.current_range = low_value, high_value

    def add_event_source(self, source):
        cb = partial(self.event_source_added, source=source)
        source.add_callback(cb)

    def source_added(self, data, source):
        self.current_samples[source] += data.shape[-1]
        self.current_times[source] = self.current_samples[source]/source.fs
        self.current_time = max(self.current_times.values())

    def event_source_added(self, data, source):
        self.current_times[source] = data[-1][1]
        self.current_time = max(self.current_times.values())


################################################################################
# Containers (defines a shared set of containers across axes)
################################################################################
class BasePlotContainer(PSIContribution):

    label = d_(Str())

    container = Typed(pg.GraphicsWidget)
    x_axis = Typed(pg.AxisItem)
    base_viewbox = Property()
    legend = Typed(pg.LegendItem)
    x_transform = Callable()

    buttons = d_(List())
    max_buttons = d_(Int(8))
    current_button = d_(Value())
    allow_auto_select = d_(Bool(True))
    auto_select = d_(Bool(True))

    @d_func
    def fmt_button(self, key):
        return str(key)

    def _observe_buttons(self, event):
        if not self.buttons:
            return
        if self.current_button not in self.buttons:
            self.current_button = self.buttons[0]

    def _observe_allow_auto_select(self, event):
        if not self.allow_auto_select:
            self.auto_select = False

    def _default_x_transform(self):
        return lambda x: x

    def _default_container(self):
        container = pg.GraphicsLayout()
        container.setSpacing(10)

        # Add the x and y axes to the layout, along with the viewbox.
        for i, child in enumerate(self.children):
            container.addItem(child.y_axis, i, 0)
            container.addItem(child.viewbox, i, 1)
            try:
                # This raises an "already taken" QGridLayoutEngine error. The
                # obvious explanation is because the current viewbox also
                # occupies this cell.
                container.addItem(child.viewbox_norm, i, 1)
            except AttributeError:
                pass

            child._configure_viewbox()

        if self.x_axis is not None:
            container.addItem(self.x_axis, i+1, 1)

        # Link the child viewboxes together
        for child in self.children[1:]:
            child.viewbox.setXLink(self.children[0].viewbox)

        return container

    def add_legend_item(self, plot, label):
        self.legend.addItem(plot, label)

    def _default_legend(self):
        legend = pg.LegendItem()
        legend.setParentItem(self.container)
        return legend

    def _get_base_viewbox(self):
        return self.children[0].viewbox

    def _default_x_axis(self):
        x_axis = pg.AxisItem('bottom')
        x_axis.setGrid(64)
        x_axis.linkToView(self.children[0].viewbox)
        return x_axis

    def update(self, event=None):
        pass

    def find(self, name):
        for child in self.children:
            if child.name == name:
                return child

    def _reset_plots(self):
        pass


class PlotContainer(BasePlotContainer):

    x_min = d_(Float(0))
    x_max = d_(Float(0))

    @observe('x_min', 'x_max')
    def format_container(self, event=None):
        # If we want to specify values relative to a psi context variable, we
        # cannot do it when initializing the plots.
        if (self.x_min != 0) or (self.x_max != 0):
            self.base_viewbox.setXRange(self.x_min, self.x_max, padding=0)

    def update(self, event=None):
        deferred_call(self.format_container)


class BaseTimeContainer(BasePlotContainer):
    '''
    Contains one or more viewboxes that share the same time-based X-axis
    '''
    data_range = Typed(BaseDataRange)
    span = d_(Float(1))
    delay = d_(Float(0.25))

    def _default_container(self):
        container = super()._default_container()
        # Ensure that the x axis shows the planned range
        self.base_viewbox.setXRange(0, self.span, padding=0)
        self.data_range.observe('current_range', self.update)
        return container

    def _default_x_axis(self):
        x_axis = super()._default_x_axis()
        x_axis.setLabel('Time', units='s')
        return x_axis

    def update(self, event=None):
        low, high = self.data_range.current_range
        deferred_call(self.base_viewbox.setXRange, low, high, padding=0)
        super().update()


class TimeContainer(BaseTimeContainer):

    def _default_data_range(self):
        return ChannelDataRange(container=self, span=self.span,
                                delay=self.delay)

    def update(self, event=None):
        for child in self.children:
            child.update()
        super().update()


class EpochTimeContainer(BaseTimeContainer):

    def _default_data_range(self):
        return EpochDataRange(container=self, span=self.span, delay=self.delay)


def format_log_ticks(values, scale, spacing):
    values = 10**np.array(values).astype(np.float)
    return ['{:.1f}'.format(v * 1e-3) for v in values]


class FFTContainer(BasePlotContainer):
    '''
    Contains one or more viewboxes that share the same frequency-based X-axis
    '''
    freq_lb = d_(Float(500))
    freq_ub = d_(Float(50000))
    octave_spacing = d_(Bool(True))

    def _default_x_transform(self):
        return np.log10

    @observe('container', 'freq_lb', 'freq_ub')
    def _update_x_limits(self, event):
        if not self.is_initialized:
            # This addresses a segfault that occurs when attempting to load
            # experiment manifests that use FFTContainer. If the Experiment
            # manifest attempts to set freq_lb or freq_ub, then it will attempt
            # to initialize everything else before the GUI is created, leading
            # to a segfault (creating an AxisItem leads to attempting to call
            # QGraphicsLabel.setHtml, which will segfault if there is no
            # instance of QtApplcation). By ensuring we don't continue if the
            # object is not initialized yet, we can properly load experiment
            # manifests (e.g., so that `psi` can properly list the available
            # paradigms).
            return
        self.base_viewbox.setXRange(np.log10(self.freq_lb),
                                    np.log10(self.freq_ub),
                                    padding=0)
        if self.octave_spacing:
            major_ticks = util.octave_space(self.freq_lb / 1e3, self.freq_ub / 1e3, 1.0)
            major_ticklabs = [str(t) for t in major_ticks]
            major_ticklocs = np.log10(major_ticks * 1e3)
            minor_ticks = util.octave_space(self.freq_lb / 1e3, self.freq_ub / 1e3, 0.125)
            minor_ticklabs = [str(t) for t in minor_ticks]
            minor_ticklocs = np.log10(minor_ticks * 1e3)
            ticks = [
                list(zip(major_ticklocs, major_ticklabs)),
                list(zip(minor_ticklocs, minor_ticklabs)),
            ]
            self.x_axis.setTicks(ticks)
        else:
            self.x_axis.setTicks()

    def _default_x_axis(self):
        x_axis = super()._default_x_axis()
        x_axis.setLabel('Frequency', units='Hz')
        x_axis.logTickStrings = format_log_ticks
        x_axis.setLogMode(True)
        return x_axis


################################################################################
# ViewBox
################################################################################
class ViewBox(PSIContribution):

    # Make this weak-referenceable so we can bind methods to Qt slots.
    __slots__ = '__weakref__'

    viewbox = Typed(pg.ViewBox)
    viewbox_norm = Typed(pg.ViewBox)
    y_axis = Typed(pg.AxisItem)

    y_min = d_(Float(0))
    y_max = d_(Float(0))
    y_mode = d_(Enum('mouse', 'fixed'))

    data_range = Property()
    save_limits = d_(Bool(False))

    @observe('y_min', 'y_max')
    def _update_limits(self, event=None):
        self.viewbox.setYRange(self.y_min, self.y_max, padding=0)

    def _default_name(self):
        return self.label

    def _get_data_range(self):
        return self.parent.data_range

    def _default_y_axis(self):
        y_axis = pg.AxisItem('left')
        y_axis.setLabel(self.label)
        y_axis.setGrid(64)
        return y_axis

    def _sync_limits(self, vb=None):
        with self.suppress_notifications():
            box = self.viewbox.viewRange()
            self.y_min = float(box[1][0])
            self.y_max = float(box[1][1])

    def _default_viewbox(self):
        return pg.ViewBox(enableMenu=False)

    def _configure_viewbox(self):
        viewbox = self.viewbox
        viewbox.setMouseEnabled(
            x=False,
            y=self.y_mode == 'mouse'
        )
        viewbox.disableAutoRange()
        viewbox.setBackgroundColor('w')
        self.y_axis.linkToView(viewbox)
        viewbox.setYRange(self.y_min, self.y_max, padding=0)

        for child in self.children:
            plots = child.get_plots()
            if isinstance(plots, dict):
                for label, plot in plots.items():
                    deferred_call(self.add_plot, plot, label)
            else:
                for plot in plots:
                    deferred_call(self.add_plot, plot)

        viewbox.sigRangeChanged.connect(self._sync_limits)
        return viewbox

    def _default_viewbox_norm(self):
        viewbox = pg.ViewBox(enableMenu=False)
        viewbox.setMouseEnabled(x=False, y=False)
        viewbox.disableAutoRange()
        return viewbox

    def update(self, event=None):
        for child in self.children:
            child.update()

    def add_plot(self, plot, label=None):
        self.viewbox.addItem(plot)
        if label:
            self.parent.legend.addItem(plot, label)

    def plot(self, x, y, color='k', log_x=False, log_y=False, label=None,
             kind='line'):
        '''
        Convenience function used by plugins

        This is typically used in post-processing routines to add static plots
        to existing view boxes.
        '''
        if log_x:
            x = np.log10(x)
        if log_y:
            y = np.log10(y)
        x = np.asarray(x)
        y = np.asarray(y)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        if kind == 'line':
            item = pg.PlotCurveItem(pen=pg.mkPen(color))
        elif kind == 'scatter':
            item = pg.ScatterPlotItem(pen=pg.mkPen(color))
        item.setData(x, y)
        self.add_plot(item)

        if label is not None:
            self.parent.legend.addItem(item, label)


################################################################################
# Plots
################################################################################
class BasePlot(PSIContribution):

    # Make this weak-referenceable so we can bind methods to Qt slots.
    __slots__ = '__weakref__'

    source_name = d_(Str())
    source = Typed(object)
    label = d_(Str())

    def update(self, event=None):
        pass

    def _reset_plots(self):
        pass


################################################################################
# Single plots
################################################################################
class SinglePlot(BasePlot):

    pen_color = d_(Typed(object))
    pen_width = d_(Float(0))
    antialias = d_(Bool(False))
    label = d_(Str())

    pen = Typed(object)
    plot = Typed(object)

    def get_plots(self):
        return [self.plot]

    def _default_pen_color(self):
        return 'black'

    def _default_pen(self):
        color = make_color(self.pen_color)
        return pg.mkPen(color, width=self.pen_width)

    def _default_name(self):
        return self.source_name + '_plot'


class ChannelPlot(SinglePlot):

    downsample = Int(0)
    decimate_mode = d_(Enum('extremes', 'mean'))

    _cached_time = Typed(np.ndarray)
    _buffer = Typed(SignalBuffer)

    def _default_name(self):
        return self.source_name + '_channel_plot'

    def _default_plot(self):
        return pg.PlotCurveItem(pen=self.pen, antialias=self.antialias)

    def _observe_source(self, event):
        if self.source is not None:
            self.parent.data_range.add_source(self.source)
            self.parent.data_range.observe('span', self._update_time)
            self.source.add_callback(self._append_data)
            self.parent.viewbox.sigResized.connect(self._update_decimation)
            self._update_time(None)
            self._update_decimation(self.parent.viewbox)

    def _update_time(self, event):
        # Precompute the time array since this can be the "slow" point
        # sometimes in computations
        n = round(self.parent.data_range.span*self.source.fs)
        self._cached_time = np.arange(n)/self.source.fs
        self._update_decimation()
        self._update_buffer()

    def _update_buffer(self, event=None):
        self._buffer = SignalBuffer(self.source.fs,
                                    self.parent.data_range.span*2)

    def _update_decimation(self, viewbox=None):
        try:
            width, _ = self.parent.viewbox.viewPixelSize()
            dt = self.source.fs**-1
            self.downsample = round(width/dt/2)
        except Exception as e:
            pass

    def _append_data(self, data):
        self._buffer.append_data(data)
        self.update()

    def update(self, event=None):
        low, high = self.parent.data_range.current_range
        data = self._buffer.get_range_filled(low, high, np.nan)
        t = self._cached_time[:len(data)] + low
        if self.downsample > 1:
            t = t[::self.downsample]
            if self.decimate_mode == 'extremes':
                d_min, d_max = decimate_extremes(data, self.downsample)
                t = t[:len(d_min)]
                x = np.c_[t, t].ravel()
                y = np.c_[d_min, d_max].ravel()
                if x.shape == y.shape:
                    deferred_call(self.plot.setData, x, y, connect='pairs')
            elif self.decimate_mode == 'mean':
                d = decimate_mean(data, self.downsample)
                t = t[:len(d)]
                if t.shape == d.shape:
                    deferred_call(self.plot.setData, t, d)
        else:
            t = t[:len(data)]
            if t.shape == data.shape:
                deferred_call(self.plot.setData, t, data)


def _reshape_for_decimate(data, downsample):
    # Determine the "fragment" size that we are unable to decimate.  A
    # downsampling factor of 5 means that we perform the operation in chunks of
    # 5 samples.  If we have only 13 samples of data, then we cannot decimate
    # the last 3 samples and will simply discard them.
    last_dim = data.ndim
    offset = data.shape[-1] % downsample
    if offset > 0:
        data = data[..., :-offset]
    shape = (len(data), -1, downsample) if data.ndim == 2 else (-1, downsample)
    return data.reshape(shape)


def decimate_mean(data, downsample):
    # If data is empty, return imediately
    if data.size == 0:
        return np.array([]), np.array([])
    data = _reshape_for_decimate(data, downsample).copy()
    return data.mean(axis=-1)


def decimate_extremes(data, downsample):
    # If data is empty, return imediately
    if data.size == 0:
        return np.array([]), np.array([])

    # Force a copy to be made, which speeds up min()/max().  Apparently min/max
    # make a copy of a reshaped array before performing the operation, so we
    # force it now so the copy only occurs once.
    data = _reshape_for_decimate(data, downsample).copy()
    return data.min(axis=-1), data.max(axis=-1)


class FFTChannelPlot(ChannelPlot):

    time_span = d_(Float(1))
    window = d_(Enum('hamming', 'flattop'))
    _x = Typed(np.ndarray)
    _buffer = Typed(SignalBuffer)

    def _default_name(self):
        return self.source_name + '_fft_plot'

    def _observe_source(self, event):
        if self.source is not None:
            self.source.add_callback(self._append_data)
            self.source.observe('fs', self._cache_x)
            self._update_buffer()
            self._cache_x()

    def _update_buffer(self, event=None):
        self._buffer = SignalBuffer(self.source.fs, self.time_span)

    def _append_data(self, data):
        self._buffer.append_data(data)
        self.update()

    def _cache_x(self, event=None):
        if self.source.fs:
            self._x = get_x_fft(self.source.fs, self.time_span)

    def update(self, event=None):
        if self._buffer.get_time_ub() >= self.time_span:
            log.trace('Time span %f to %f', -self.time_span, 0)
            data = self._buffer.get_latest(-self.time_span, 0)
            psd = util.psd(data, self.source.fs, self.window)
            db = self.source.calibration.get_db(self._x, psd)
            if self._x.shape == db.shape:
                deferred_call(self.plot.setData, self._x, db)


class BaseTimeseriesPlot(SinglePlot):

    rect_center = d_(Float(0.5))
    rect_height = d_(Float(1))
    fill_color = d_(Typed(object))
    brush = Typed(object)
    _rising = Typed(list, ())
    _falling = Typed(list, ())

    def _default_brush(self):
        return pg.mkBrush(self.fill_color)

    def _default_plot(self):
        plot = pg.QtGui.QGraphicsPathItem()
        plot.setPen(self.pen)
        plot.setBrush(self.brush)
        return plot

    def update(self, event=None):
        lb, ub = self.parent.data_range.current_range
        current_time = self.parent.data_range.current_time

        starts = self._rising
        ends = self._falling
        if len(starts) == 0 and len(ends) == 1:
            starts = [0]
        elif len(starts) == 1 and len(ends) == 0:
            ends = [current_time]
        elif len(starts) > 0 and len(ends) > 0:
            if starts[0] > ends[0]:
                starts = np.r_[0, starts]
            if starts[-1] > ends[-1]:
                ends = np.r_[ends, current_time]

        try:
            epochs = np.c_[starts, ends]
        except ValueError as e:
            log.exception(e)
            log.warning('Unable to update %r, starts shape %r, ends shape %r',
                        self, starts, ends)
            return

        m = ((epochs >= lb) & (epochs < ub)) | np.isnan(epochs)
        epochs = epochs[m.any(axis=-1)]

        path = pg.QtGui.QPainterPath()
        y_start = self.rect_center - self.rect_height*0.5
        for x_start, x_end in epochs:
            x_width = x_end-x_start
            r = pg.QtCore.QRectF(x_start, y_start, x_width, self.rect_height)
            path.addRect(r)

        deferred_call(self.plot.setPath, path)


class EventPlot(BaseTimeseriesPlot):

    event = d_(Str())

    def _observe_event(self, event):
        if self.event is not None:
            self.parent.data_range.observe('current_time', self.update)

    def _default_name(self):
        return self.event + '_timeseries'

    def _append_data(self, bound, timestamp):
        if bound == 'start':
            self._rising.append(timestamp)
        elif bound == 'end':
            self._falling.append(timestamp)
        self.update()


class TimeseriesPlot(BaseTimeseriesPlot):

    source_name = d_(Str())
    source = Typed(object)

    def _default_name(self):
        return self.source_name + '_timeseries'

    def _observe_source(self, event):
        if self.source is not None:
            self.parent.data_range.add_event_source(self.source)
            self.parent.data_range.observe('current_time', self.update)
            self.source.add_callback(self._append_data)

    def _append_data(self, data):
        for (etype, value) in data:
            if etype == 'rising':
                self._rising.append(value)
            elif etype == 'falling':
                self._falling.append(value)


################################################################################
# Group plots
################################################################################
class FixedTextItem(pg.TextItem):

    def updateTransform(self, force=False):
        p = self.parentItem()
        if p is None:
            pt = pg.QtGui.QTransform()
        else:
            pt = p.sceneTransform()

        if not force and pt == self._lastTransform:
            return

        t = pt.inverted()[0]
        # reset translation
        t.setMatrix(1, t.m12(), t.m13(), t.m21(), 1, t.m23(), 0, 0, t.m33())

        # apply rotation
        angle = -self.angle
        if self.rotateAxis is not None:
            d = pt.map(self.rotateAxis) - pt.map(Point(0, 0))
            a = np.arctan2(d.y(), d.x()) * 180 / np.pi
            angle += a
        t.rotate(angle)

        self.setTransform(t)
        self._lastTransform = pt
        self.updateTextPos()


class GroupMixin(ColorCycleMixin):

    source = Typed(object)

    pen_width = d_(Int(0))
    antialias = d_(Bool(False))

    plots = Dict()
    labels = Dict()

    _data_cache = Typed(object)
    _data_count = Typed(object)
    _data_updated = Typed(object)
    _data_n_samples = Typed(object)

    _pen_color_cycle = Typed(object)
    _plot_colors = Typed(object)
    _x = Typed(np.ndarray)

    n_update = d_(Int(1))

    #: List of attributes that define the tab groups
    tab_grouping = d_(List())

    #: List of attributes that define the plot groups
    plot_grouping = d_(List())

    #: List of existing tab keys
    tab_keys = d_(List())

    #: List of existing plot keys
    plot_keys = d_(List())

    #: Which tab is currently selected?
    selected_tab = d_(Value())

    #: Should we auto-select the tab based on the most recently acquired data?
    auto_select = d_(Bool(False))

    #: What was the most recent tab key seen?
    last_seen_key = Value()

    #: Function that takes the epoch metadata and returns a key that is used to
    #: assign the epoch to a group. Return None to exclude the epoch from the
    #: group criteria.
    @d_func
    def group_key(self, md):
        plot_key = tuple(md[a] for a in self.plot_grouping)
        tab_key = tuple(md[a] for a in self.tab_grouping)
        return tab_key, plot_key

    @d_func
    def fmt_plot_label(self, key):
        return None

    def _default_selected_tab(self):
        return ()

    def _observe_selected_tab(self, event):
        self.update(tab_changed=True)

    @observe('last_seen_key', 'auto_select')
    def _update_selected_tab(self, event):
        if not self.auto_select:
            return
        if self.last_seen_key is None:
            return
        if self.last_seen_key[0] != self.selected_tab:
            self.selected_tab = self.last_seen_key[0]

    def _reset_plots(self):
        # Clear any existing plots and reset color cycle
        for plot in self.plots.items():
            self.parent.viewbox.removeItem(plot)
        for label in self.labels.items():
            self.parent.viewbox_norm.removeItem(label)
        self.plots = {}
        self._data_cache = defaultdict(list)
        self._data_count = defaultdict(int)
        self._data_updated = defaultdict(int)
        self._data_n_samples = defaultdict(int)

    def get_plots(self):
        return []

    def _make_new_plot(self, key):
        try:
            pen_color = self.get_pen_color(key)
            pen = pg.mkPen(pen_color, width=self.pen_width)
            plot = pg.PlotCurveItem(pen=pen, antialias=self.antialias)
            self.plots[key] = plot
            deferred_call(self.parent.viewbox.addItem, plot)

            label = self.fmt_plot_label(key)
            if label is not None:
                text = pg.TextItem(label, color=pen_color,
                                   border=pg.mkPen(pen_color),
                                   fill=pg.mkBrush('w'))
                deferred_call(self.parent.viewbox_norm.addItem, text)
                self.labels[key] = text
        except KeyError as key_error:
            key = key_error.args[0]
            m = f'Cannot update plot since a field, {key}, ' \
                 'required by the plot is missing.'
            raise ConfigurationException(m) from key_error

    def get_plot(self, key):
        if key not in self.plots:
            self._make_new_plot(key)
        return self.plots[key]


class EpochGroupMixin(GroupMixin):

    duration = Float()

    def _y(self, epoch):
        return np.mean(epoch, axis=0) if len(epoch) \
            else np.full_like(self._x, np.nan)

    def _update_duration(self, event=None):
        self.duration = self.source.duration

    def _epochs_acquired(self, epochs):
        for d in epochs:
            key = self.group_key(d['info']['metadata'])
            if key is not None:
                signal = d['signal']
                self._data_cache[key].append(signal)
                self._data_count[key] += 1

                # Track number of samples
                n = max(self._data_n_samples[key], len(signal))
                self._data_n_samples[key] = n

        self.last_seen_key = key

        # Does at least one epoch need to be updated?
        self._check_selected_tab_count()

    def _get_selected_tab_keys(self):
        return [k for k in self._data_count if k[0] == self.selected_tab]

    def _check_selected_tab_count(self):
        for key in self._get_selected_tab_keys():
            current_n = self._data_count[key]
            last_n = self._data_updated[key]
            if current_n >= (last_n + self.n_update):
                n = max(self._data_n_samples.values())
                self.duration = n / self.source.fs
                self.update()
                break

    def _observe_source(self, event):
        if self.source is not None:
            self.source.add_callback(self._epochs_acquired)
            self.source.observe('duration', self._update_duration)
            self.source.observe('fs', self._cache_x)
            self.observe('duration', self._cache_x)
            self._reset_plots()
            self._cache_x()

    def _observe_selected_tab(self, event):
        self.update(tab_changed=True)

    def update(self, event=None, tab_changed=False):
        todo = []
        if self._x is None:
            return
        for pk in self.plot_keys:
            plot = self.get_plot(pk)
            key = (self.selected_tab, pk)
            try:
                last_n = self._data_updated[key]
                current_n = self._data_count[key]
                needs_update = current_n >= (last_n + self.n_update)
                if tab_changed or needs_update:
                    data = self._data_cache[key]
                    self._data_updated[key] = len(data)
                    if data:
                        x = self._x
                        y = self._y(data)
                    else:
                        x = y = np.array([])
                    if x.shape == y.shape:
                        todo.append((plot.setData, x, y))
            except KeyError:
                if tab_changed:
                    x = y = np.array([])
                    todo.append((plot.setData, x, y))

        def update():
            for setter, x, y in todo:
                setter(x, y)
        deferred_call(update)


class GroupedEpochAveragePlot(EpochGroupMixin, BasePlot):

    def _cache_x(self, event=None):
        # Set up the new time axis
        if self.source.fs and self.duration:
            n_time = round(self.source.fs * self.duration)
            self._x = np.arange(n_time)/self.source.fs

    def _default_name(self):
        return self.source_name + '_grouped_epoch_average_plot'

    def _observe_source(self, event):
        super()._observe_source(event)
        if self.source is not None:
            self.parent.data_range.add_source(self.source)


class GroupedEpochFFTPlot(EpochGroupMixin, BasePlot):

    waveform_averages = d_(Int(1))

    def _default_name(self):
        return self.source_name + '_grouped_epoch_fft_plot'

    def _cache_x(self, event=None):
        # Cache the frequency points. Must be in units of log for PyQtGraph.
        # TODO: This could be a utility function stored in the parent?
        if self.source.fs and self.duration:
            self._x = get_x_fft(self.source.fs, self.duration / self.waveform_averages)

    def _y(self, epoch):
        y = np.mean(epoch, axis=0) if epoch else np.full_like(self._x, np.nan)
        psd = util.psd(y, self.source.fs, waveform_averages=self.waveform_averages)
        return self.source.calibration.get_db(self._x, psd)


class GroupedEpochPhasePlot(EpochGroupMixin, BasePlot):

    unwrap = d_(Bool(True))

    def _default_name(self):
        return self.source_name + '_grouped_epoch_phase_plot'

    def _cache_x(self, event=None):
        # Cache the frequency points. Must be in units of log for PyQtGraph.
        # TODO: This could be a utility function stored in the parent?
        if self.source.fs and self.duration:
            self._x = get_x_fft(self.source.fs, self.duration)

    def _y(self, epoch):
        y = np.mean(epoch, axis=0) if epoch else np.full_like(self._x, np.nan)
        return util.phase(y, self.source.fs, unwrap=self.unwrap)


class StackedEpochAveragePlot(EpochGroupMixin, BasePlot):

    _offset_update_needed = Bool(False)

    def _make_new_plot(self, key):
        super()._make_new_plot(key)
        self._offset_update_needed = True

    def _update_offsets(self, vb=None):
        vb = self.parent.viewbox
        height = vb.height()
        n = len(self.plots)

        plot_items = sorted(self.plots.items(), reverse=True)
        for i, (key, plot) in enumerate(plot_items):
            offset = (i+1) * height / (n+1)
            point = self.parent.viewbox.mapToView(pg.Point(0, offset))
            plot.setPos(0, point.y())

        labels = sorted(self.labels.items(), reverse=True)
        for i, (key, label) in enumerate(labels):
            offset = (i+1) * height / (n+1)
            point = self.parent.viewbox_norm.mapToView(pg.Point(0, offset))
            label.setPos(0.8, point.y())

    def _cache_x(self, event=None):
        # Set up the new time axis
        if self.source.fs and self.source.duration:
            n_time = round(self.source.fs * self.source.duration)
            self._x = np.arange(n_time)/self.source.fs

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        if self._offset_update_needed:
            deferred_call(self._update_offsets)
            self._offset_update_needed = False

    def _reset_plots(self):
        super()._reset_plots()
        self.parent.viewbox \
            .sigRangeChanged.connect(self._update_offsets)
        self.parent.viewbox \
            .sigRangeChangedManually.connect(self._update_offsets)


################################################################################
# Simple plotters
################################################################################
class ResultPlot(GroupMixin, SinglePlot):

    x_column = d_(Str())
    y_column = d_(Str())
    average = d_(Bool())

    SYMBOL_MAP = {
        'circle': 'o',
        'square': 's',
        'triangle': 't',
        'diamond': 'd',
    }
    symbol = d_(Enum('circle', 'square', 'triangle', 'diamond'))
    symbol_size = d_(Float(10))
    symbol_size_unit = d_(Enum('screen', 'data'))

    def get_plots(self):
        return {self.label: self.plot}

    def _default_name(self):
        return '.'.join((self.parent.name, self.source_name, 'result_plot',
                         self.x_column, self.y_column))

    def _observe_source(self, event):
        if self.source is not None:
            self._data_cache = {}
            self.source.add_callback(self._data_acquired)

    def _data_acquired(self, data):
        for d in data:
            key = self.group_key(d)
            if key is not None:
                cache = self._data_cache.setdefault(key, {'x': [], 'y': []})
                cache['x'].append(d[self.x_column])
                cache['y'].append(d[self.y_column])
        self.last_seen_key = key
        self.update()

    def update(self, event=None, tab_changed=False):
        default = {'x': [], 'y': []}
        key = (self.selected_tab, ())
        data = self._data_cache.get(key, default)
        x = np.array(data['x'])
        y = np.array(data['y'])
        if self.average:
            d = pd.DataFrame({'x': x, 'y': y}).groupby('x')['y'].mean()
            x = d.index.values
            y = d.values
        if x.shape == y.shape:
            deferred_call(self.plot.setData, x, y)

    def _default_plot(self):
        symbol_code = self.SYMBOL_MAP[self.symbol]
        color = QColor(self.pen_color)
        pen = pg.mkPen(color, width=self.pen_width)
        brush = pg.mkBrush(color)
        return pg.PlotDataItem(pen=pen, antialias=self.antialias,
                               symbol=symbol_code, symbolSize=self.symbol_size,
                               symbolPen=pen, symbolBrush=brush,
                               pxMode=self.symbol_size_unit=='screen')


class DataFramePlot(ColorCycleMixin, PSIContribution):

    data = d_(Typed(pd.DataFrame))
    x_column = d_(Str())
    y_column = d_(Str())
    grouping = d_(List(Str()))
    _plot_cache = Dict()

    SYMBOL_MAP = {
        'circle': 'o',
        'square': 's',
        'triangle': 't',
        'diamond': 'd',
    }
    symbol = d_(Enum('circle', 'square', 'triangle', 'diamond'))
    symbol_size = d_(Float(10))
    symbol_size_unit = d_(Enum('screen', 'data'))

    pen_width = d_(Float(0))
    antialias = d_(Bool(False))

    container = Property()

    def _get_container(self):
        parent = self.parent
        while True:
            if isinstance(parent, BasePlotContainer):
                return parent
            parent = parent.parent

    def _default_name(self):
        return '.'.join((self.parent.name, 'result_plot'))

    def _observe_x_column(self, event):
        self._reset_plots()
        self._observe_data(event)

    def _observe_y_column(self, event):
        self._reset_plots()
        self._observe_data(event)

    def _observe_grouping(self, event):
        self._reset_plots()
        self._observe_data(event)

    def _observe_data(self, event):
        if self.data is None:
            return
        if self.x_column not in self.data:
            return
        if self.y_column not in self.data:
            return

        todo = []
        if self.grouping:
            try:
                for group, values in self.data.groupby(self.grouping):
                    if len(self.grouping) == 1:
                        label = str(group)
                    else:
                        label = ','.join(f'{n} {v}'
                                        for n, v in zip(self.grouping, group))
                    if group not in self._plot_cache:
                        self._plot_cache[group] = self._make_plot(group, label)
                    x = values[self.x_column].values
                    y = values[self.y_column].values
                    x = self.container.x_transform(x)
                    i = np.argsort(x)
                    todo.append((self._plot_cache[group], x[i], y[i]))
            except KeyError as e:
                # This is likely triggered when grouping updates an analysis
                # before it's ready.
                log.warning(e)
                return
        else:
            if None not in self._plot_cache:
                self._plot_cache[None] = self._make_plot(None)
            x = self.data[self.x_column].values
            y = self.data[self.y_column].values
            x = self.container.x_transform(x)
            i = np.argsort(x)
            todo.append((self._plot_cache[None], x[i], y[i]))

        def update():
            nonlocal todo
            for plot, x, y in todo:
                plot.setData(x, y)
        deferred_call(update)

    def _make_plot(self, group, label=None):
        symbol_code = self.SYMBOL_MAP[self.symbol]
        color = self.get_pen_color(group)
        brush = pg.mkBrush(color)
        pen = pg.mkPen(color, width=self.pen_width)

        plot = pg.PlotDataItem(pen=pen,
                               antialias=self.antialias,
                               symbol=symbol_code,
                               symbolSize=self.symbol_size,
                               symbolPen=pen,
                               symbolBrush=brush,
                               pxMode=self.symbol_size_unit=='screen')
        deferred_call(self.parent.add_plot, plot, label)
        return plot

    def _reset_plots(self):
        for plot in self._plot_cache.values():
            deferred_call(self.parent.viewbox.removeItem, plot)
        self._plot_cache = {}

    def get_plots(self):
        return list(self._plot_cache.values())
