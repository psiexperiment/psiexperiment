import logging
log = logging.getLogger(__name__)

import threading
from functools import partial
import struct

import numpy as np
import pyqtgraph as pg

from atom.api import (Unicode, Float, Tuple, Int, Typed, Property, Atom, Bool, Enum)
from enaml.core.api import Declarative, d_
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

        # Add the x and y axes to the layout
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

    freq_lb = d_(Float(5))
    freq_ub = d_(Float(50000))

    def _default_container(self):
        container = super()._default_container()
        self.base_viewbox.setXRange(np.log10(self.freq_lb), np.log10(self.freq_ub), padding=0)
        return container

    def _default_x_axis(self):
        x_axis = super()._default_x_axis()
        x_axis.setLabel('Frequency', unitPrefix='Hz')
        x_axis.logTickStrings = format_log_ticks
        x_axis.setLogMode(True)
        return x_axis


################################################################################
# Plots
################################################################################
class CustomGraphicsViewBox(pg.ViewBox):

    def __init__(self, data_range, *args, **kwargs):
        self.data_range = data_range
        super().__init__(*args, **kwargs)

    def wheelEvent(self, ev, axis=None):
        s = 1.02**(ev.delta() * self.state['wheelScaleFactor'])
        if axis == 0:
            self.data_range.span *= s
        elif axis == 1:
            vr = self.targetRect()
            y_max = vr.topLeft().y() * s
            y_min = vr.bottomRight().y() * s
            self.setYRange(y_min, y_max)
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
    y_min = d_(Float())
    y_max = d_(Float())

    data_range = Property()

    def _get_data_range(self):
        return self.parent.data_range

    def _default_y_axis(self):
        y_axis = pg.AxisItem('left')
        y_axis.linkToView(self.viewbox)
        y_axis.setGrid(64)
        return y_axis

    def _default_viewbox(self):
        viewbox = CustomGraphicsViewBox(self.parent.data_range,
                                        enableMenu=False)
        viewbox.setBackgroundColor('w')
        viewbox.disableAutoRange()
        viewbox.setYRange(self.y_min, self.y_max)
        for child in self.children:
            viewbox.addItem(child.plot)
        return viewbox

    def _update(self, event=None):
        for child in self.children:
            child._update()


class Plot(PSIContribution):

    pen_color = d_(Typed(object))
    pen_width = d_(Float(0))
    antialias = d_(Bool(False))
    label = d_(Unicode())

    pen = Typed(object)
    plot = Typed(object)

    update_pending = Bool(False)

    def _default_pen_color(self):
        return 'k'

    def _default_pen(self):
        return pg.mkPen(self.pen_color, width=self.pen_width)

    def prepare(self, plugin):
        pass

    def update(self, event=None):
        if not self.update_pending:
            deferred_call(self._update, event)
            self.update_pending = True
        else:
            log.debug('Update already pending for %s', self.source_name)

    def _update(self, event=None):
        self.update_pending = False


class ChannelPlot(Plot):

    source_name = d_(Unicode())
    source = Typed(object)

    downsample = Int(0)
    _cached_time = Typed(np.ndarray)

    def _default_plot(self):
        return pg.PlotCurveItem(pen=self.pen, antialias=self.antialias)

    def _observe_source(self, event):
        self.parent.data_range.add_source(self.source, self)
        self.parent.data_range.observe('span', self.update_time)
        self.parent.viewbox.sigResized.connect(lambda vb: self.update_decimation())
        self.update_time(None)
        self.update_decimation(self.parent.viewbox)

    def update_time(self, event):
        # Precompute the time array since this can be the "slow" point
        # sometimes in computations
        n = int(self.parent.data_range.span*self.source.fs)
        self._cached_time = np.arange(n)/self.source.fs
        self.update_decimation()

    def update_decimation(self, event=None):
        try:
            width, _ = self.parent.viewbox.viewPixelSize()
            dt = self.source.fs**-1
            self.downsample = int(width/dt/5)
        except Exception as e:
            pass

    def _update(self, event=None):
        low, high = self.parent.data_range.current_range
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

    def _default_plot(self):
        return pg.PlotCurveItem(pen=self.pen, antialias=self.antialias)

    def prepare(self, plugin):
        self.source = plugin.find_source(self.source_name)
        n = int(self.source.fs*self.time_span)
        self._cached_frequency = np.log10(np.fft.rfftfreq(n, self.source.fs**-1))
        self.source.observe('added', self.update)

    def _update(self, event=None):
        ub = event['value']['ub']
        if ub < self.time_span:
            return
        data = self.source.get_range(ub-self.time_span, ub)
        psd = util.patodb(util.psd(data, self.source.fs))
        self.plot.setData(self._cached_frequency, psd)
        self.update_pending = False


class TimeseriesPlot(Plot):

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


class EpochAveragePlot(ChannelPlot):

    filters = Typed(dict, {})

    def _observe_source(self, event):
        n = int(self.parent.data_range.span*self.source.fs)
        self._cached_time = np.arange(n)/self.source.fs
        self.update_decimation(self.parent.viewbox)
        self.source.observe('added', self.update)

    def _update(self, event=None):
        filters = {p.name: v for p, v in self.filters.items()}
        result = self.source.get_epochs(filters)
        if len(result) == 0:
            return
        y = result.mean(axis=0)
        self.plot.setData(self._cached_time, y)
        self.update_pending = False
