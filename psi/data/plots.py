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
    current_time = Float()
    current_range = Tuple(Float(), Float())

    def _default_current_range(self):
        return 0, self.span

    def add_source(self, source, plot):
        source.observe('added', partial(self.source_added, plot=plot))

    def source_added(self, event, plot):
        self.current_time = max(self.current_time, event['value']['ub'])
        low_value = (self.current_time//self.span)*self.span
        high_value = low_value+self.span
        if self.current_range != (low_value, high_value):
            # this updates all plots linked to this data range
            self.current_range = (low_value, high_value)
            self.container.update_range(low_value, high_value)
        else:
            # this updates only the plot that has new data
            plot.update_range(low_value, high_value)


################################################################################
# Containers (defines a shared set of containers across axes)
################################################################################
class PlotContainer(PSIContribution):

    title = d_(Unicode())
    label = d_(Unicode())
    container = Typed(pg.GraphicsWidget)

    x_axis = Typed(pg.AxisItem)
    base_viewbox = Property()

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
        x_axis.setGrid(127)
        x_axis.linkToView(self.children[0].viewbox)
        return x_axis

    def prepare(self, plugin):
        for child in self.children:
            child.prepare(plugin)


class TimeContainer(PlotContainer):

    data_range = Typed(ChannelDataRange)
    span = d_(Float(1))
    delay = d_(Float(0.25))

    pixel_width = Float()

    def _default_container(self):
        container = super()._default_container()
        # Ensure that the x axis shows the planned range
        self.base_viewbox.setXRange(0, self.span, padding=0)
        return container

    def _default_data_range(self):
        return ChannelDataRange(container=self, span=self.span)

    def update_range(self, low, high):
        for child in self.children:
            child.update_range(low, high)
        self.base_viewbox.setXRange(low, high, padding=0)

    def _default_x_axis(self):
        x_axis = super()._default_x_axis()
        x_axis.setLabel('Time', unitPrefix='sec.')
        return x_axis


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
        y_axis.setGrid(127)
        return y_axis

    def _default_viewbox(self):
        viewbox = pg.ViewBox(enableMouse=True, enableMenu=False)
        viewbox.setBackgroundColor('w')
        viewbox.disableAutoRange()
        viewbox.setYRange(self.y_min, self.y_max)
        for child in self.children:
            viewbox.addItem(child.plot)
        viewbox.sigResized.connect(lambda vb: child.update_decimation(vb))
        return viewbox

    def prepare(self, plugin):
        for child in self.children:
            child.prepare(plugin)

    def update_range(self, low, high):
        for child in self.children:
            child.update_range(low, high)


class Plot(PSIContribution):

    pen_color = d_(Typed(object))
    pen_width = d_(Float(0))
    antialias = d_(Bool(False))

    pen = Typed(object)
    plot = Typed(object)

    def _default_pen(self):
        return pg.mkPen(self.pen_color, width=self.pen_width)

    def update_decimation(self, vb):
        pass

    def prepare(self, plugin):
        pass


class ChannelPlot(Plot):

    source_name = d_(Unicode())
    source = Typed(object)

    downsample = Int(0)
    _cached_time = Typed(np.ndarray)

    def _default_plot(self):
        return pg.PlotCurveItem(pen=self.pen, antialias=self.antialias)

    def prepare(self, plugin):
        self.source = plugin.find_source(self.source_name)
        self.parent.data_range.add_source(self.source, self)

        # Precompute the time array since this can be the "slow" point
        # sometimes in computations
        n = int(self.parent.data_range.span*self.source.fs)
        self._cached_time = np.arange(n)/self.source.fs
        self.update_decimation(self.parent.viewbox)

    def update_decimation(self, vb):
        try:
            width, _ = vb.viewPixelSize()
            dt = self.source.fs**-1
            self.downsample = int(width/dt/10)
        except Exception as e:
            pass

    def update_range(self, low, high):
        data = self.source.get_range(low, high)
        t = self._cached_time[:len(data)] + low
        if self.downsample > 1:
            t = t[::self.downsample]
            d_min, d_max = decimate_extremes(data, self.downsample)
            t = t[:len(d_min)]
            x = np.c_[t, t].ravel()
            y = np.c_[d_min, d_max].ravel()
            deferred_call(self.plot.setData, x, y, connect='pairs')
        else:
            deferred_call(self.plot.setData, t, data)


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

    # Force a copy to be made, which speeds up min()/max().  Apparently min/max
    # make a copy of a reshaped array before performing the operation, so we
    # force it now so the copy only occurs once.
    if data.ndim == 2:
        shape = (len(data), -1, downsample)
    else:
        shape = (-1, downsample)
    data = data[..., :-offset].reshape(shape).copy()
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

    def update(self, event):
        print(self.plot.isVisible())
        ub = event['value']['ub']
        if ub < self.time_span:
            return
        data = self.source.get_range(ub-self.time_span, ub)
        psd = util.patodb(util.psd(data, self.source.fs))
        deferred_call(self.plot.setData, self._cached_frequency, psd)


class TimeseriesPlot(Plot):

    source_name = d_(Unicode())
    rising_event = d_(Unicode())
    falling_event = d_(Unicode())
    rect_center = d_(Float(0.5))
    rect_height = d_(Float(1))

    fill_color = d_(Typed(object))

    brush = Typed(object)
    source = Typed(object)

    def _default_brush(self):
        return pg.mkBrush(self.fill_color)

    def _default_plot(self):
        plot = pg.QtGui.QGraphicsPathItem()
        plot.setPen(self.pen)
        plot.setBrush(self.brush)
        return plot
        
    def prepare(self, plugin):
        self.source = plugin.find_source(self.source_name)
        self.parent.data_range.add_source(self.source, self)
        self.parent.data_range.observe('current_time', self.update)

    def update(self, event=None):
        low, high = self.parent.data_range.current_range
        current_time = self.parent.data_range.current_time
        epochs = self.source.get_epochs(self.rising_event, self.falling_event,
                                        low, high, current_time)

        path = pg.QtGui.QPainterPath()
        y_start = self.rect_center - self.rect_height*0.5
        for x_start, x_end in epochs:
            x_width = x_end-x_start
            r = pg.QtCore.QRectF(x_start, y_start, x_width, self.rect_height)
            path.addRect(r)
        deferred_call(self.plot.setPath, path)

    def update_range(self, low, high):
        self.update()


################################################################################
# Special case. Need to think this through.
################################################################################
class GridContainer(PlotContainer):

    items = d_(Typed(dict))
    source_name = d_(Unicode())
    update_delay = d_(Float(0))

    source = Typed(object)

    trial_log = Typed(object)

    cumulative_average = Typed(dict, {})
    cumulative_var = Typed(dict, {})
    cumulative_n = Typed(dict, {})

    grid = Typed(tuple)
    plots = Typed(dict)
    time = Typed(object)

    context_info = Typed(object)

    _update_pending = Bool(False)

    def _default_container(self):
        return pg.GraphicsLayout()

    def context_info_updated(self, info):
        self.context_info = info

    def _default_grid(self):
        return set(), set()

    def prepare(self, plugin):
        self.source = plugin.find_source(self.source_name)
        self.source.observe('added', self.epochs_acquired)
        self.cumulative_average = {}
        self.cumulative_var = {}
        self.cumulative_n = {}

    def prepare_grid(self, iterable):
        self.items = {
            'row': {'name': 'target_tone_level'},
            'column': {'name': 'target_tone_frequency'},
        }
        keys = [self.extract_key(c) for c in iterable]
        self.update_grid(keys)

    def extract_key(self, context):
        ci = self.items['row']
        row = None if ci is None else context[ci['name']]
        ci = self.items['column']
        column = None if ci is None else context[ci['name']]
        return row, column

    def epochs_acquired(self, event):
        for epoch in event['value']:
            self.process_epoch(epoch)
        if self.update_delay > 0:
            if not self._update_pending:
                self._update_pending = True
                delay_ms = int(self.update_delay*1e3)
                timed_call(delay_ms, self.update_plots)
        else:
            deferred_call(self.update_plots)

    def process_epoch(self, epoch):
        key = self.extract_key(epoch['metadata'])
        signal = epoch['signal']

        n = self.cumulative_n.get(key, 0) + 1
        time, average = self.cumulative_average.get(key, (None, signal))
        delta = signal-average
        average = average + delta/n

        # TODO: Would be nice to move this out (e.g., precompute)?
        if time is None:
            time = np.arange(len(average))/self.source.fs

        self.cumulative_average[key] = time, average
        self.cumulative_n[key] = n

    def update_grid(self, keys):
        rows, cols = zip(*keys)
        rows = set(rows)
        cols = set(cols)
        cur_rows, cur_cols = self.grid
        if rows.issubset(cur_rows) and cols.issubset(cur_cols):
            return

        log.debug('Updating grid')
        self.container.clear()
        plots = {}
        base_item = None
        for c, col in enumerate(sorted(cols)):
            self.container.addLabel(col, 0, c+1)
        for r, row in enumerate(sorted(rows)):
            self.container.addLabel(row, r+1, 0, angle=-90)

        for r, row in enumerate(sorted(rows)):
            for c, col in enumerate(sorted(cols)):
                viewbox = CustomViewBox()
                item = pg.PlotItem(viewBox=viewbox)
                self.container.addItem(item, r+1, c+1)
                if base_item is None:
                    base_item = item
                else:
                    item.setXLink(base_item)
                    item.setYLink(base_item)

                item.hideButtons()
                if c != 0:
                    item.hideAxis('left')
                if r != (len(rows)-1):
                    item.hideAxis('bottom')

                item.setMouseEnabled(x=False, y=True)
                item.setXRange(0, 8.5e-3)
                item.setYRange(-0.5, 0.5)
                plots[row, col] = item.plot(pen='k')

        self.plots = plots
        self.grid = (rows, cols)

    def update_plots(self):
        keys = self.cumulative_average.keys()
        self.update_grid(keys)
        for k, (time, average) in list(self.cumulative_average.items()):
            self.plots[k].setData(time, average)
        self._update_pending = False
