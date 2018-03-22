import logging
log = logging.getLogger(__name__)

import itertools
from functools import partial
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
# Supporting classes
################################################################################
class ChannelDataRange(Atom):

    container = Typed(object)
    span = Float(1)
    delay = Float(0)
    current_time = Float(0)
    current_range = Tuple(Float(), Float())

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
        source.observe('added', partial(self.source_added, plot=plot))

    def source_added(self, event, plot):
        self.current_time = max(self.current_time, event['value']['ub'])
        plot.update()


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
        return ChannelDataRange(container=self, span=self.span)

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
        x_axis.setLabel('Frequency', unitPrefix='Hz')
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
        self.setYRange(self.y_min, self.y_max)

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
        viewbox.disableAutoRange()
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


class SignalBuffer:

    def __init__(self, fs, size):
        self._lock = threading.RLock()
        self._buffer_fs = fs
        self._buffer_size = size
        self._buffer_samples = int(fs*size)
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
            ilb = int(lb*self._buffer_fs) - self._offset
            iub = int(ub*self._buffer_fs) - self._offset
            if ilb < 0:
                raise ValueError
            return self._buffer[ilb:iub]


class ChannelPlot(SinglePlot):

    downsample = Int(0)
    _cached_time = Typed(np.ndarray)
    _buffer = Typed(SignalBuffer)

    def _default_plot(self):
        return pg.PlotCurveItem(pen=self.pen, antialias=self.antialias)

    def _observe_source(self, event):
        self.parent.data_range.add_source(self.source, self)
        self.parent.data_range.observe('span', self._update_time)
        self.source.observe('added', self._append_data)
        self.parent.viewbox.sigResized.connect(self._update_decimation)
        self._update_time(None)
        self._update_decimation(self.parent.viewbox)

    def _update_time(self, event):
        # Precompute the time array since this can be the "slow" point
        # sometimes in computations
        n = int(self.parent.data_range.span*self.source.fs)
        self._cached_time = np.arange(n)/self.source.fs
        self._update_decimation()
        self._update_buffer()

    def _update_buffer(self, event=None):
        self._buffer = SignalBuffer(self.source.fs,
                                    self.parent.data_range.span*5)

    def _update_decimation(self, viewbox=None):
        try:
            width, _ = self.parent.viewbox.viewPixelSize()
            dt = self.source.fs**-1
            self.downsample = int(width/dt/5)
        except Exception as e:
            pass

    def _append_data(self, event):
        self._buffer.append_data(event['value']['data'])

    def _update(self, event=None):
        low, high = self.parent.data_range.current_range
        #data = self._buffer.get_range(low, high)
        data = self.source.get_range(low, high)
        t = self._cached_time[:len(data)] + low
        if self.downsample > 1:
            t = t[::self.downsample]
            d_min, d_max = decimate_extremes(data, self.downsample)
            t = t[:len(d_min)]
            x = np.c_[t, t].ravel()
            y = np.c_[d_min, d_max].ravel()
            self.plot.setData(x, y, connect='pairs')
        else:
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
    _cached_frequency = Typed(np.ndarray)

    def _observe_source(self, event):
        if self.source is None:
            return
        self.source.observe('added', self.update)

        n_time = int(self.source.fs*self.time_span)
        freq = np.fft.rfftfreq(n_time, self.source.fs**-1)
        self._cached_frequency = np.log10(freq)

    def _update(self, event=None):
        ub = event['value']['ub']
        if ub >= self.time_span:
            data = self.source.get_range(ub-self.time_span, ub)
            psd = util.patodb(util.psd(data, self.source.fs))
            self.plot.setData(self._cached_frequency, psd)
        self.update_pending = False


class TimeseriesPlot(SinglePlot):

    source_name = d_(Unicode())
    rising_event = d_(Unicode())
    falling_event = d_(Unicode())
    rect_center = d_(Float(0.5))
    rect_height = d_(Float(1))

    fill_color = d_(Typed(object))

    brush = Typed(object)
    source = Typed(object)

    _rising = Typed(list, ())
    _falling = Typed(list, ())

    def _default_name(self):
        return self.source_name + self.rising_event + '_timeseries'

    def _default_brush(self):
        return pg.mkBrush(self.fill_color)

    def _default_plot(self):
        plot = pg.QtGui.QGraphicsPathItem()
        plot.setPen(self.pen)
        plot.setBrush(self.brush)
        return plot

    def _observe_source(self, event):
        self.parent.data_range.add_source(self.source, self)
        self.parent.data_range.observe('current_time', self.update)
        self.source.observe('added', self.added)

    def added(self, event):
        value = event['value']
        if value['event'] == self.rising_event:
            self._rising.append(value['lb'])
        elif value['event'] == self.falling_event:
            self._falling.append(value['lb'])

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


# TODO: Can we eliminate these in favor of the Grouped plots as they're a
# special case of a grouped plot?
class EpochAveragePlot(ChannelPlot):

    n_epochs = Int()

    def _observe_source(self, event):
        if self.source is None:
            return
        n = int(self.parent.data_range.span*self.source.fs)
        self._cached_time = np.arange(n)/self.source.fs
        self.update_decimation(self.parent.viewbox)
        self.source.observe('added', self.update)

    def _update(self, event=None):
        if self.source is None:
            return
        result = self._get_epochs()
        y = result.mean(axis=0) if len(result) \
            else np.zeros_like(self._cached_time)
        self.plot.setData(self._cached_time, y)
        self.n_epochs = len(result)
        self.update_pending = False

    def _get_epochs(self):
        return self.source.get_epochs()


class FFTEpochPlot(ChannelPlot):

    n_time = Int(0)
    n_epochs = Int()
    _cached_frequency = Typed(np.ndarray)

    def _observe_source(self, event):
        if self.source is None:
            return
        self.source.observe('added', self.update)

        # Cache the frequency points. Must be in units of log for PyQtGraph.
        n_time = int(self.source.fs * self.source.epoch_size)
        freq = np.fft.rfftfreq(n_time, self.source.fs**-1)
        self._cached_frequency = np.log10(freq)

    def _get_epochs(self):
        return self.source.get_epochs()

    def _update(self, event=None):
        if self.source is None:
            return
        epoch = self._get_epochs()
        if len(epoch):
            y = epoch.mean(axis=0)
            psd = util.db(util.psd(y, self.source.fs))
        else:
            psd = np.full_like(self._cached_frequency, np.nan)
        self.plot.setData(self._cached_frequency, psd)
        self.update_pending = False


################################################################################
# Group plots
################################################################################
class GroupOverlayMixin(Declarative):

    groups = d_(List())
    pen_color_cycle = d_(List())

    group_names = List()
    plots = Dict()

    _pen_color_cycle = Typed(object)

    def _default_pen_color_cycle(self):
        return ['k']

    @d_func
    def get_pen_color(self, key):
        return next(self._pen_color_cycle)

    def _reset_plots(self):
        # Clear any existing plots
        for plot in self.plots.items():
            self.parent.viewbox.removeItem(plot)
        self.plots = {}

        # Set up the color cycle
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


class GroupedEpochAveragePlot(BasePlot, GroupOverlayMixin):

    _cached_time = Typed(np.ndarray)

    def _observe_source(self, event):
        if self.source is None:
            return
        self._reset_plots()

        # Set up the new time axis
        n = int(self.parent.data_range.span*self.source.fs)
        self._cached_time = np.arange(n)/self.source.fs

        # Subscribe to notifications
        self.source.observe('added', self.update)

    def _update(self, event=None):
        epochs = self.source.get_epoch_groups(self.group_names)
        for key, epoch in epochs.items():
            plot = self.get_plot(key)
            y = epoch.mean(axis=0) if len(epoch) \
                else np.full_like(self._cached_time, np.nan)
            plot.setData(self._cached_time, y)
        self.update_pending = False


class GroupedEpochFFTPlot(BasePlot, GroupOverlayMixin):

    _cached_frequency = Typed(np.ndarray)

    def _observe_source(self, event):
        if self.source is None:
            return
        self._reset_plots()

        # Cache the frequency points. Must be in units of log for PyQtGraph.
        # TODO: This could be a utility function stored in the parent?
        n_time = int(self.source.fs * self.source.epoch_size)
        freq = np.fft.rfftfreq(n_time, self.source.fs**-1)
        self._cached_frequency = np.log10(freq)

        # Subscribe to notifications
        self.source.observe('added', self.update)

    def _update(self, event=None):
        epochs = self.source.get_epoch_groups(self.group_names)
        for key, epoch in epochs.items():
            plot = self.get_plot(key)
            y = epoch.mean(axis=0) if len(epoch) \
                else np.full_like(self._cached_frequency, np.nan)
            psd = util.db(util.psd(y, self.source.fs))
            plot.setData(self._cached_frequency, psd)
        self.update_pending = False


class StackedEpochAveragePlot(BasePlot, GroupOverlayMixin):

    _cached_time = Typed(np.ndarray)

    def _make_new_plot(self, key):
        super()._make_new_plot(key)
        self._update_offsets()

    def _observe_source(self, event):
        if self.source is None:
            return
        self._reset_plots()

        # Set up the new time axis
        n = int(self.parent.data_range.span*self.source.fs)
        self._cached_time = np.arange(n)/self.source.fs

        # Subscribe to notifications
        self.source.observe('added', self.update)

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

    def _update(self, event=None):
        if self.source is None:
            return
        epochs = self.source.get_epoch_groups(self.group_names)
        for key, epoch in epochs.items():
            plot = self.get_plot(key)
            y = epoch.mean(axis=0) if len(epoch) \
                else np.full_like(self._cached_time, np.nan)
            plot.setData(self._cached_time, y)
        self.update_pending = False