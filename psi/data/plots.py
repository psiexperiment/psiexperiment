import logging

log = logging.getLogger(__name__)

import itertools
import importlib
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
import pyqtgraph as pg

from atom.api import (Unicode, Float, Tuple, Int, Typed, Property, Atom, Bool,
                      Enum, List, Dict, Callable, Value)

from enaml.application import deferred_call, timed_call
from enaml.colors import parse_color
from enaml.core.api import Looper, Declarative, d_, d_func
from enaml.core.compiler_nodes import new_scope
from enaml.qt.QtGui import QColor

from psi.util import SignalBuffer, ConfigurationException
from psi.core.enaml.api import load_manifests, PSIContribution
from psi.controller.calibration import util
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


def create_container(children, x_axis=None):
    container = pg.GraphicsLayout()
    container.setSpacing(10)

    # Add the x and y axes to the layout, along with the viewbox.
    for i, child in enumerate(children):
        container.addItem(child.y_axis, i, 0)
        container.addItem(child.viewbox, i, 1)

    if x_axis is not None:
        container.addItem(x_axis, i+1, 1)

    # Link the child viewboxes together
    for child in children[1:]:
        child.viewbox.setXLink(children[0].viewbox)

    #children[0].viewbox.setXRange(0, 100, padding=0)
    return container


################################################################################
# Pattern containers
################################################################################
from enaml.core.pattern import Pattern


class MultiPlotContainer(Pattern, PSIContribution):

    group = d_(Unicode())
    selected_item = Value()

    workbench = Value()
    iterable = List()
    items = List()
    containers = d_(Dict())

    def pattern_items(self):
        return sum(self.items, [])

    def refresh_items(self):
        if (not self.iterable) or (len(self.pattern_nodes) == 0):
            return
        items = []
        log.debug('Iterable is %r', self.iterable)
        for loop_index, loop_item in enumerate(self.iterable):
            for nodes, key, f_locals in self.pattern_nodes:
                f_locals['loop_index'] = loop_index
                f_locals['loop_item'] = loop_item
                for node in nodes:
                    with new_scope(key, f_locals) as f_locals:
                        child = node(None)
                        if isinstance(child, list):
                            items.extend(child)
                        else:
                            items.append(child)
        self.items = items
        load_manifests(self.items, self.workbench)
        self.containers = {str(i): c.container for i, c in zip(self.iterable, self.items)}
        log.debug(self.containers)


################################################################################
# Containers (defines a shared set of containers across axes)
################################################################################
class BasePlotContainer(PSIContribution):

    label = d_(Unicode())

    container = Typed(pg.GraphicsWidget)
    x_axis = Typed(pg.AxisItem)
    base_viewbox = Property()
    legend = Typed(pg.LegendItem)
    x_transform = Callable()

    def _default_x_transform(self):
        return lambda x: x

    def _default_container(self):
        return create_container(self.children, self.x_axis)

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

    def format_container(self):
        pass

    def _reset_plots(self):
        pass


class PlotContainer(BasePlotContainer):

    x_min = d_(Float(0))
    x_max = d_(Float(0))

    def format_container(self):
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
        x_axis.setLabel('Time', unitPrefix='sec.')
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
    return ['{:.1f}'.format(v) for v in values]


class FFTContainer(BasePlotContainer):
    '''
    Contains one or more viewboxes that share the same frequency-based X-axis
    '''
    freq_lb = d_(Float(5))
    freq_ub = d_(Float(50000))

    def _default_x_transform(self):
        return np.log10

    def _default_container(self):
        container = super()._default_container()
        self.base_viewbox.setXRange(np.log10(self.freq_lb),
                                    np.log10(self.freq_ub),
                                    padding=0)
        return container

    def _default_x_axis(self):
        x_axis = super()._default_x_axis()
        x_axis.setLabel('Frequency (Hz)')
        x_axis.logTickStrings = format_log_ticks
        x_axis.setLogMode(True)
        return x_axis


################################################################################
# ViewBox
################################################################################
class ViewBox(PSIContribution):

    viewbox = Typed(pg.ViewBox)
    y_axis = Typed(pg.AxisItem)

    y_mode = d_(Enum('symmetric', 'upper'))

    y_min = d_(Float(0))
    y_max = d_(Float(0))

    allow_zoom_y = d_(Bool(True))
    allow_zoom_x = d_(Bool(False))

    data_range = Property()

    def _default_name(self):
        return self.label

    def _get_data_range(self):
        return self.parent.data_range

    def _default_y_axis(self):
        y_axis = pg.AxisItem('left')
        y_axis.setLabel(self.label)
        y_axis.linkToView(self.viewbox)
        y_axis.setGrid(64)
        return y_axis

    def _default_viewbox(self):
        viewbox = pg.ViewBox(enableMenu=False)
        viewbox.setMouseEnabled(x=False, y=True)
        viewbox.setBackgroundColor('w')

        if (self.y_min != 0) or (self.y_max != 0):
            viewbox.disableAutoRange()
            viewbox.setYRange(self.y_min, self.y_max)

        for child in self.children:
            for plot in child.get_plots():
                viewbox.addItem(plot)
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

    source_name = d_(Unicode())
    source = Typed(object)
    label = d_(Unicode())

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
    label = d_(Unicode())

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
            data = self._buffer.get_latest(-self.time_span, 0)
            #psd = util.patodb(util.psd(data, self.source.fs, self.window))
            psd = util.psd(data, self.source.fs, self.window)
            spl = self.source.calibration.get_spl(self._x, psd)
            deferred_call(self.plot.setData, self._x, spl)


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

    event = d_(Unicode())

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

    source_name = d_(Unicode())
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
class GroupMixin(ColorCycleMixin):

    source = Typed(object)
    group_meta = d_(Unicode())
    groups = d_(Typed(ContextMeta))
    group_names = d_(List())

    #: Function that takes the epoch metadata and decides whether to accept it
    #: for plotting.  Useful to reduce the number of plots shown on a graph.
    group_filter = d_(Callable())

    #: Function that takes the epoch metadata and returns a key indicating
    #: which group it should included in for plotting.
    group_color_key = d_(Callable())

    pen_width = d_(Int(0))
    antialias = d_(Bool(False))

    plots = Dict()

    _data_cache = Typed(object)
    _data_count = Typed(object)
    _data_updated = Typed(object)
    _data_n_samples = Typed(object)

    _pen_color_cycle = Typed(object)
    _plot_colors = Typed(object)
    _x = Typed(np.ndarray)

    n_update = d_(Int(1))

    def _default_group_names(self):
        return [p.name for p in self.groups.values]

    def _default_group_filter(self):
        return lambda key: True

    def _default_group_color_key(self):
        return lambda key: tuple(key[g] for g in self.group_names)

    def get_pen_color(self, key):
        kw_key = {n: k for n, k in zip(self.group_names, key)}
        group_key = self.group_color_key(kw_key)
        return super().get_pen_color(group_key)

    def _reset_plots(self):
        # Clear any existing plots and reset color cycle
        for plot in self.plots.items():
            self.parent.viewbox.removeItem(plot)
        self.plots = {}
        self._data_cache = defaultdict(list)
        self._data_count = defaultdict(int)
        self._data_updated = defaultdict(int)
        self._data_n_samples = defaultdict(int)

    def _observe_groups(self, event):
        self.groups.observe('values', self._update_groups)
        self._update_groups()

    def _update_groups(self, event=None):
        self._reset_plots()
        self.group_names = [p.name for p in self.groups.values]
        if self.source is not None:
            self.update()

    def get_plots(self):
        return []

    def _make_new_plot(self, key):
        log.info('Adding plot for key %r', key)
        try:
            pen_color = self.get_pen_color(key)
            pen = pg.mkPen(pen_color, width=self.pen_width)
            plot = pg.PlotCurveItem(pen=pen, antialias=self.antialias)
            deferred_call(self.parent.viewbox.addItem, plot)
            self.plots[key] = plot
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
            md = d['info']['metadata']
            if self.group_filter(md):
                signal = d['signal']
                key = tuple(md[n] for n in self.group_names)
                self._data_cache[key].append(signal)
                self._data_count[key] += 1

                # Track number of samples
                n = max(self._data_n_samples[key], len(signal))
                self._data_n_samples[key] = n

        # Does at least one epoch need to be updated?
        for key, count in self._data_count.items():
            if count >= self._data_updated[key] + self.n_update:
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

    def update(self, event=None):
        # Update epochs that need updating
        todo = []
        for key, count in list(self._data_count.items()):
            if count >= self._data_updated[key] + self.n_update:
                data = self._data_cache[key]
                plot = self.get_plot(key)
                y = self._y(data)
                todo.append((plot.setData, self._x, y))
                self._data_updated[key] = len(data)

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

    def _default_name(self):
        return self.source_name + '_grouped_epoch_fft_plot'

    def _cache_x(self, event=None):
        # Cache the frequency points. Must be in units of log for PyQtGraph.
        # TODO: This could be a utility function stored in the parent?
        if self.source.fs and self.duration:
            self._x = get_x_fft(self.source.fs, self.duration)

    def _y(self, epoch):
        if not epoch:
            return np.full_like(self._x, np.nan)
        y = np.mean(epoch, axis=0)
        return self.source.calibration.get_spl(self._x, util.psd(y, self.source.fs))


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
        if not epoch:
            return np.full_like(self._x, np.nan)
        y = np.mean(epoch, axis=0)
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
        for i, (_, plot) in enumerate(sorted(self.plots.items())):
            offset = (i+1) * height / (n+1)
            point = vb.mapToView(pg.Point(0, offset))
            plot.setPos(0, point.y())

    def _cache_x(self, event=None):
        # Set up the new time axis
        if self.source.fs and self.source.duration:
            n_time = round(self.source.fs * self.source.duration)
            self._x = np.arange(n_time)/self.source.fs

    def update(self):
        super().update()
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
class ResultPlot(SinglePlot):

    x_column = d_(Unicode())
    y_column = d_(Unicode())
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

    data_filter = d_(Callable())

    _data_cache = Typed(list)

    def _default_data_filter(self):
        # By default, accept all data points
        return lambda x: True

    def _default_name(self):
        return '.'.join((self.parent.name, self.source_name, 'result_plot',
                         self.x_column, self.y_column))

    def _observe_source(self, event):
        if self.source is not None:
            self._data_cache = []
            self.source.add_callback(self._data_acquired)

    def _data_acquired(self, data):
        update = False
        for d in data:
            if self.data_filter(d):
                x = d[self.x_column]
                y = d[self.y_column]
                self._data_cache.append((x, y))
                update = True
        if update:
            self.update()

    def update(self, event=None):
        if not self._data_cache:
            return
        x, y = zip(*self._data_cache)
        x = np.array(x)
        y = np.array(y)
        if self.average:
            d = pd.DataFrame({'x': x, 'y': y}).groupby('x')['y'].mean()
            x = d.index.values
            y = d.values

        deferred_call(self.plot.setData, x, y)

    def _default_plot(self):
        symbol_code = self.SYMBOL_MAP[self.symbol]
        color = QColor(self.pen_color)
        pen = pg.mkPen(color, width=self.pen_width)
        brush = pg.mkBrush(color)

        plot = pg.PlotDataItem(pen=pen,
                                antialias=self.antialias,
                                symbol=symbol_code,
                                symbolSize=self.symbol_size,
                                symbolPen=pen,
                                symbolBrush=brush,
                                pxMode=self.symbol_size_unit=='screen')

        deferred_call(self.parent.add_plot, plot, self.label)
        return plot


class DataFramePlot(ColorCycleMixin, PSIContribution):

    data = d_(Typed(pd.DataFrame))
    x_column = d_(Unicode())
    y_column = d_(Unicode())
    grouping = d_(List(Unicode()))
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
                        self._plot_cache[group] = self._default_plot(group, label)
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
                self._plot_cache[None] = self._default_plot(None)
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

    def _default_plot(self, group, label=None):
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
