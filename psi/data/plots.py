import logging

log = logging.getLogger(__name__)

import itertools
from functools import partial
from collections import defaultdict
import threading

import numpy as np
import pyqtgraph as pg

from atom.api import (Unicode, Float, Tuple, Int, Typed, Property, Atom, Bool,
                      Enum, List, Dict)
from enaml.core.api import Declarative, d_, d_func
from enaml.application import deferred_call, timed_call

from psi.core.enaml.api import PSIContribution
from psi.controller.calibration import util


################################################################################
# Utility functions
################################################################################
def get_x_fft(fs, duration):
    n_time = int(fs * duration)
    freq = np.fft.rfftfreq(n_time, fs**-1)
    return np.log10(freq)


class SignalBuffer:

    def __init__(self, fs, size):
        self._lock = threading.RLock()
        self._buffer_fs = fs
        self._buffer_size = size
        self._buffer_samples = round(fs*size)
        self._buffer = np.full(self._buffer_samples, np.nan)
        self._offset = -self._buffer_samples

    def append_data(self, data):
        with self._lock:
            samples = data.shape[-1]
            if samples > self._buffer_samples:
                self._buffer[:] = data[-self.buffer_samples:]
            else:
                self._buffer[:-samples] = self._buffer[samples:]
                self._buffer[-samples:] = data
            self._offset += samples

    def get_range(self, lb, ub):
        with self._lock:
            ilb = round(lb*self._buffer_fs) - self._offset
            iub = round(ub*self._buffer_fs) - self._offset
            if ilb < 0:
                raise ValueError
            return self._buffer[ilb:iub]

    def get_latest(self, lb, ub=0):
        with self._lock:
            lb = lb + self.get_time_ub()
            ub = ub + self.get_time_ub()
            return self.get_range(lb, ub)

    def get_time_ub(self):
        return (self._offset + self._buffer_samples) / self._buffer_fs

    def get_time_lb(self):
        return max(self._offset, 0) / self._buffer_fs

    def get_valid_data(self):
        with self._lock:
            return self._buffer[:]


################################################################################
# Supporting classes
################################################################################
class ChannelDataRange(Atom):

    container = Typed(object)
    span = Float(1)
    delay = Float(0)
    current_time = Float(0)
    current_range = Tuple(Float(), Float())

    current_samples = Typed(defaultdict, (int,))
    current_times = Typed(defaultdict, (float,))

    def _default_current_range(self):
        return 0, self.span

    def _observe_delay(self, event):
        self._update_range()

    def _observe_current_time(self, event):
        self._update_range()

    def _observe_span(self, event):
        self._update_range()

    def _update_range(self):
        low_value = (self.current_time//self.span)*self.span - self.delay
        high_value = low_value+self.span
        self.current_range = low_value, high_value

    def add_source(self, source, plot):
        cb = partial(self.source_added, plot=plot, fs=source.fs)
        source.add_callback(cb)

    def add_event_source(self, source, plot):
        cb = partial(self.event_source_added, plot=plot, fs=source.fs)
        source.add_callback(cb)

    def source_added(self, data, plot, fs):
        self.current_samples[plot] += data.shape[-1]
        self.current_times[plot] = self.current_samples[plot]/fs
        self.current_time = max(self.current_times.values())

    def event_source_added(self, data, plot, fs):
        self.current_times[plot] = data[-1][1]
        self.current_time = max(self.current_times.values())


################################################################################
# Containers (defines a shared set of containers across axes)
################################################################################
class PlotContainer(PSIContribution):

    label = d_(Unicode())
    label = d_(Unicode())
    container = Typed(pg.GraphicsWidget)
    x_axis = Typed(pg.AxisItem)
    base_viewbox = Property()
    update_pending = Bool(False)
    legend = Typed(pg.LegendItem)

    def _default_container(self):
        container = pg.GraphicsLayout()
        container.setSpacing(10)

        # Add the x and y axes to the layout, along with the viewbox.
        for i, child in enumerate(self.children):
            container.addItem(child.y_axis, i, 0)
            container.addItem(child.viewbox, i, 1)
        container.addItem(self.x_axis, i+1, 1)

        # Link the child viewboxes together
        for child in self.children[1:]:
            child.viewbox.setXLink(self.base_viewbox)

        return container

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
        if not self.update_pending:
            deferred_call(self._update, event)
            self.update_pending = True

    def _update(self, event=None):
        self.update_pending = False

    def find(self, name):
        for child in self.children:
            if child.name == name:
                return child


class TimeContainer(PlotContainer):
    '''
    Contains one or more viewboxes that share the same time-based X-axis
    '''
    data_range = Typed(ChannelDataRange)
    span = d_(Float(1))
    delay = d_(Float(0.25))

    pixel_width = Float()

    def _default_container(self):
        container = super()._default_container()
        # Ensure that the x axis shows the planned range
        self.base_viewbox.setXRange(0, self.span, padding=0)
        self.data_range.observe('current_range', self.update)
        return container

    def _default_data_range(self):
        return ChannelDataRange(container=self, span=self.span,
                                delay=self.delay)

    def _default_x_axis(self):
        x_axis = super()._default_x_axis()
        x_axis.setLabel('Time', unitPrefix='sec.')
        return x_axis

    def _update(self, event=None):
        low, high = self.data_range.current_range
        current_time = self.data_range.current_time
        for child in self.children:
            child._update()
        self.base_viewbox.setXRange(low, high, padding=0)
        super()._update()


def format_log_ticks(values, scale, spacing):
    values = 10**np.array(values).astype(np.float)
    return ['{:.1f}'.format(v) for v in values]


class FFTContainer(PlotContainer):
    '''
    Contains one or more viewboxes that share the same frequency-based X-axis
    '''
    freq_lb = d_(Float(5))
    freq_ub = d_(Float(50000))

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
class CustomGraphicsViewBox(pg.ViewBox):

    def __init__(self, data_range, y_min, y_max, y_mode, allow_zoom_x,
                 allow_zoom_y, *args, **kwargs):
        self.data_range = data_range
        self.y_min = y_min
        self.y_max = y_max
        self.y_mode = y_mode
        self.allow_zoom_x = allow_zoom_x
        self.allow_zoom_y = allow_zoom_y
        super().__init__(*args, **kwargs)

    def wheelEvent(self, ev, axis=None):
        if axis == 0 and not self.allow_zoom_x:
            return
        if axis == 1 and not self.allow_zoom_y:
            return

        s = 1.02**(ev.delta() * self.state['wheelScaleFactor'])

        if axis == 0:
            self.data_range.span *= s
        elif axis == 1:
            vr = self.targetRect()
            if self.y_mode == 'symmetric':
                self.y_min *= s
                self.y_max *= s
            elif self.y_mode == 'upper':
                self.y_max *= s
            self.setYRange(self.y_min, self.y_max)

        self.sigRangeChangedManually.emit(self.state['mouseEnabled'])
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        ev.accept()
        return
        delta = ev.pos()-ev.lastPos()
        tr = self.mapToView(delta)-self.mapToView(pg.Point(0, 0))
        if axis == 0:
            x = tr.x()
            self.data_range.delay += x
        ev.accept()


class ViewBox(PSIContribution):

    viewbox = Typed(pg.ViewBox)
    y_axis = Typed(pg.AxisItem)

    y_mode = d_(Enum('symmetric', 'upper'))
    y_min = d_(Float())
    y_max = d_(Float())

    allow_zoom_y = d_(Bool(True))
    allow_zoom_x = d_(Bool(False))

    data_range = Property()

    def _get_data_range(self):
        return self.parent.data_range

    def _default_y_axis(self):
        y_axis = pg.AxisItem('left')
        y_axis.setLabel(self.label)
        y_axis.linkToView(self.viewbox)
        y_axis.setGrid(64)
        return y_axis

    def _default_viewbox(self):
        try:
            viewbox = CustomGraphicsViewBox(self.parent.data_range,
                                            self.y_min,
                                            self.y_max,
                                            self.y_mode,
                                            self.allow_zoom_x,
                                            self.allow_zoom_y,
                                            enableMenu=False)
        except:
            viewbox = pg.ViewBox(enableMenu=False)
        viewbox.setBackgroundColor('w')

        if (self.y_min != 0) or (self.y_max != 0):
            viewbox.disableAutoRange()
            viewbox.setYRange(self.y_min, self.y_max)

        for child in self.children:
            for plot in child.get_plots():
                viewbox.addItem(plot)
        return viewbox

    def _update(self, event=None):
        for child in self.children:
            child._update()

    def add_plot(self, plot):
        self.viewbox.addItem(plot)


################################################################################
# Plots
################################################################################
class BasePlot(PSIContribution):

    # Make this weak-referenceable so we can bind methods to Qt slots.
    __slots__ = '__weakref__'

    source_name = d_(Unicode())
    source = Typed(object)
    update_pending = Bool(False)
    label = d_(Unicode())

    def update(self, event=None):
        if not self.update_pending:
            deferred_call(self._update, event)
            self.update_pending = True

    def _update(self, event=None):
        raise NotImplementedError


################################################################################
# Single plots
################################################################################
class SinglePlot(BasePlot):

    pen_color = d_(Typed(object))
    pen_width = d_(Float(0))
    antialias = d_(Bool(False))

    pen = Typed(object)
    plot = Typed(object)

    def get_plots(self):
        return [self.plot]

    def _default_pen_color(self):
        return 'k'

    def _default_pen(self):
        return pg.mkPen(self.pen_color, width=self.pen_width)


class ChannelPlot(SinglePlot):

    downsample = Int(0)
    _cached_time = Typed(np.ndarray)
    _buffer = Typed(SignalBuffer)

    def _default_plot(self):
        return pg.PlotCurveItem(pen=self.pen, antialias=self.antialias)

    def _observe_source(self, event):
        if self.source is not None:
            self.parent.data_range.add_source(self.source, self)
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
        self._buffer = SignalBuffer(self.source.fs, self.parent.data_range.span*2)

    def _update_decimation(self, viewbox=None):
        try:
            width, _ = self.parent.viewbox.viewPixelSize()
            dt = self.source.fs**-1
            self.downsample = round(width/dt/5)
        except Exception as e:
            pass

    def _append_data(self, data):
        self._buffer.append_data(data)
        self.update()

    def _update(self, event=None):
        low, high = self.parent.data_range.current_range
        data = self._buffer.get_range(low, high)
        t = self._cached_time[:len(data)] + low
        if self.downsample > 1:
            t = t[::self.downsample]
            d_min, d_max = decimate_extremes(data, self.downsample)
            t = t[:len(d_min)]
            x = np.c_[t, t].ravel()
            y = np.c_[d_min, d_max].ravel()
            self.plot.setData(x, y, connect='pairs')
        else:
            t = t[:len(data)]
            self.plot.setData(t, data)
        self.update_pending = False


def decimate_extremes(data, downsample):
    # If data is empty, return imediately
    if data.size == 0:
        return np.array([]), np.array([])

    # Determine the "fragment" size that we are unable to decimate.  A
    # downsampling factor of 5 means that we perform the operation in chunks of
    # 5 samples.  If we have only 13 samples of data, then we cannot decimate
    # the last 3 samples and will simply discard them.
    last_dim = data.ndim
    offset = data.shape[-1] % downsample
    if offset > 0:
        data = data[..., :-offset]

    # Force a copy to be made, which speeds up min()/max().  Apparently min/max
    # make a copy of a reshaped array before performing the operation, so we
    # force it now so the copy only occurs once.
    if data.ndim == 2:
        shape = (len(data), -1, downsample)
    else:
        shape = (-1, downsample)
    data = data.reshape(shape).copy()
    return data.min(last_dim), data.max(last_dim)


class FFTChannelPlot(ChannelPlot):

    time_span = d_(Float())
    window = d_(Enum('hamming', 'flattop'))
    _x = Typed(np.ndarray)
    _buffer = Typed(SignalBuffer)

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

    def _update(self, event=None):
        if self._buffer.get_time_ub() >= self.time_span:
            data = self._buffer.get_latest(-self.time_span, 0)
            psd = util.patodb(util.psd(data, self.source.fs, self.window))
            self.plot.setData(self._x, psd)
        self.update_pending = False


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

    def _update(self, event=None):
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

        epochs = np.c_[starts, ends]
        m = ((epochs >= lb) & (epochs < ub)) | np.isnan(epochs)
        epochs = epochs[m.any(axis=-1)]

        path = pg.QtGui.QPainterPath()
        y_start = self.rect_center - self.rect_height*0.5
        for x_start, x_end in epochs:
            x_width = x_end-x_start
            r = pg.QtCore.QRectF(x_start, y_start, x_width, self.rect_height)
            path.addRect(r)
        self.plot.setPath(path)
        self.update_pending = False


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
            self.parent.data_range.add_event_source(self.source, self)
            self.parent.data_range.observe('current_time', self.update)
            self.source.add_callback(self._append_data)

    def _append_data(self, data):
        print(data)
        #value = event['value']
        #if value['event'] == self.rising_event:
        #    self._rising.append(value['lb'])
        #elif value['event'] == self.falling_event:
        #    self._falling.append(value['lb'])

################################################################################
# Group plots
################################################################################
class GroupMixin(Declarative):

    source = Typed(object)
    groups = d_(List())
    pen_color_cycle = d_(List())

    group_names = List()
    plots = Dict()

    _epoch_cache = Typed(object)
    _epoch_count = Typed(object)
    _epoch_updated = Typed(object)
    _pen_color_cycle = Typed(object)
    _x = Typed(np.ndarray)

    n_update = d_(Int(1))

    def _epochs_acquired(self, epochs):
        for d in epochs:
            md = d['info']['metadata']
            signal = d['signal']
            key = tuple(md[n] for n in self.group_names)
            self._epoch_cache[key].append(signal)
            self._epoch_count[key] += 1

        # Does at least one epoch need to be updated?
        for key, count in self._epoch_count.items():
            if count >= self._epoch_updated[key] + self.n_update:
                self.update()
                break

    def _default_pen_color_cycle(self):
        return ['k']

    @d_func
    def get_pen_color(self, key):
        return next(self._pen_color_cycle)

    def _reset_plots(self):
        # Clear any existing plots and reset color cycle
        for plot in self.plots.items():
            self.parent.viewbox.removeItem(plot)
        self.plots = {}
        self._epoch_cache = defaultdict(list)
        self._epoch_count = defaultdict(int)
        self._epoch_updated = defaultdict(int)
        self._pen_color_cycle = itertools.cycle(self.pen_color_cycle)

    def _observe_groups(self, event):
        self._reset_plots()
        self.group_names = [p.name for p in self.groups]
        if self.source is not None:
            self.update()

    def _observe_pen_color_cycle(self, event):
        self._reset_plots()

    def get_plots(self):
        return []

    def _make_new_plot(self, key):
        log.info('Adding plot for key %r', key)
        pen = pg.mkPen(self.get_pen_color(key))
        plot = pg.PlotCurveItem(pen=pen)
        self.parent.viewbox.addItem(plot)
        self.plots[key] = plot

    def get_plot(self, key):
        if key not in self.plots:
            self._make_new_plot(key)
        return self.plots[key]

    def _y(self, epoch):
        return np.mean(epoch, axis=0) if len(epoch) \
            else np.full_like(self._x, np.nan)

    def _update(self, event=None):
        # Update epochs that need updating
        for key, count in list(self._epoch_count.items()):
            if count >= self._epoch_updated[key] + self.n_update:
                epoch = self._epoch_cache[key]
                plot = self.get_plot(key)
                y = self._y(epoch)
                plot.setData(self._x, y)
                self._epoch_updated[key] = len(epoch)
        self.update_pending = False

    def _observe_source(self, event):
        if self.source is not None:
            self.source.add_callback(self._epochs_acquired)
            self.source.observe('fs', self._cache_x)
            self.source.observe('epoch_size', self._cache_x)
            self._reset_plots()

    def _cache_x(self, event):
        # Set up the new time axis
        if self.source.fs and self.source.epoch_size:
            n_time = round(self.source.fs * self.source.epoch_size)
            self._x = np.arange(n_time)/self.source.fs


class GroupedEpochAveragePlot(GroupMixin, BasePlot):
    pass


class GroupedEpochFFTPlot(GroupMixin, BasePlot):

    def _cache_x(self, event):
        # Cache the frequency points. Must be in units of log for PyQtGraph.
        # TODO: This could be a utility function stored in the parent?
        if self.source.fs and self.source.epoch_size:
            self._x = get_x_fft(self.source.fs, self.source.epoch_size)

    def _y(self, epoch):
        y = np.mean(epoch, axis=0) if epoch else np.full_like(self._x, np.nan)
        return util.db(util.psd(y, self.source.fs))


class StackedEpochAveragePlot(GroupMixin, BasePlot):

    def _make_new_plot(self, key):
        super()._make_new_plot(key)
        self._update_offsets()

    def _update_offsets(self, vb=None):
        vb = self.parent.viewbox
        height = vb.height()
        n = len(self.plots)
        for i, (_, plot) in enumerate(sorted(self.plots.items())):
            offset = (i+1) * height / (n+1)
            point = vb.mapToView(pg.Point(0, offset))
            plot.setPos(0, point.y())

    def _reset_plots(self):
        super()._reset_plots()
        self.parent.viewbox \
            .sigRangeChanged.connect(self._update_offsets)
        self.parent.viewbox \
            .sigRangeChangedManually.connect(self._update_offsets)
