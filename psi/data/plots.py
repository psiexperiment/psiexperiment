import logging
log = logging.getLogger(__name__)

from functools import partial
import struct

import numpy as np
import pyqtgraph as pg

from atom.api import (Unicode, Float, Tuple, Int, Typed, Property, Atom, Bool, Enum)
from enaml.core.api import Declarative, d_
from enaml.application import deferred_call, timed_call

from psi.core.enaml.api import PSIContribution


################################################################################
# Supporting classes
################################################################################
class ChannelDataRange(object):

    def __init__(self, container, span, delay=0.25):
        self.container = container
        self.span = span
        self.delay = delay
        self.current_time = -delay
        self.current_range = None

    def add_source(self, source, plot):
        source.observe('added', partial(self.source_added, plot=plot))

    def source_added(self, event, plot):
        self.current_time = max(self.current_time, event['value']['ub']-self.delay)
        spans = self.current_time // self.span
        high_value = (spans+1)*self.span
        low_value = high_value-self.span
        if high_value == low_value:
            return
        elif self.current_range != (low_value, high_value):
            self.container.update_range(low_value, high_value)
        else:
            plot.update_range(low_value, high_value)

    def update_range(self, range):
        self.current_range = (low_value, high_value)
        self.container.update_range(low_value, high_value)


################################################################################
# Containers (defines a shared set of containers across axes)
################################################################################
class PlotContainer(PSIContribution):

    title = d_(Unicode())
    label = d_(Unicode())
    container = Typed(pg.GraphicsWidget)

    def _default_container(self):
        return pg.GraphicsLayout()


class TimeContainer(PlotContainer):

    data_range = Typed(ChannelDataRange)
    span = d_(Float(1))
    delay = d_(Float(0.25))

    x_axis = Typed(pg.AxisItem)
    base_viewbox = Property()

    def _get_base_viewbox(self):
        return self.children[0].viewbox

    def _default_x_axis(self):
        x_axis = pg.AxisItem('bottom')
        x_axis.setGrid(127)
        x_axis.setLabel('Time', unitPrefix='sec.')
        x_axis.linkToView(self.children[0].viewbox)
        return x_axis

    def _default_container(self):
        container = super()._default_container()
        container.setSpacing(10)

        # Add the axes to the layout
        for i, child in enumerate(self.children):
            container.addItem(child.y_axis, i, 0)
            container.addItem(child.viewbox, i, 1)
        container.addItem(self.x_axis, i+1, 1)

        # Link the child views
        #container.addItem(self.base_viewbox, 0, i+1)
        for child in self.children[1:]:
            child.viewbox.setXLink(self.base_viewbox)

        return container

    def prepare(self, plugin):
        self.data_range = ChannelDataRange(self, self.span)
        for child in self.children:
            child.create_plot(plugin, self)
        self.update_range(-self.span, self.delay)

    def update_range(self, low, high):
        updaters = [child.update_range(low, high) for child in self.children]
        self.base_viewbox.setXRange(low, high, padding=0)
        deferred_call(lambda c=updaters: [u() for u in c])


################################################################################
# Plots
################################################################################
class Plot(PSIContribution):
    line_color = d_(Tuple())
    fill_color = d_(Tuple())


class ChannelPlot(Plot):

    source_name = d_(Unicode())
    value_range = d_(Tuple(Float(), Float()))

    y_axis = Typed(pg.AxisItem)
    viewbox = Typed(pg.ViewBox)
    source = Typed(object)
    plot = Typed(object)
    pen = Typed(object)


class ExtremesChannelPlot(ChannelPlot):

    y_min = d_(Float())
    y_max = d_(Float())
    pen_color = d_(Typed(object))
    pen_width = d_(Float(0))

    downsample_method = d_(Enum('peak', 'mean'))
    auto_downsample = d_(Bool(True))
    antialias = d_(Bool(True))

    def _default_y_axis(self):
        y_axis = pg.AxisItem('left')
        y_axis.setPen(self.pen_color)
        y_axis.linkToView(self.viewbox)
        y_axis.setGrid(127)
        return y_axis

    def _default_viewbox(self):
        viewbox = pg.ViewBox(enableMouse=True, enableMenu=False)
        viewbox.setBackgroundColor('w')
        viewbox.disableAutoRange()
        viewbox.setYRange(self.y_min, self.y_max)
        viewbox.addItem(self.plot)
        return viewbox

    def _default_pen(self):
        return pg.mkPen(self.pen_color, width=self.pen_width)

    def _default_plot(self):
        return pg.PlotCurveItem(pen=self.pen, antialias=self.antialias,
                                downsampleMethod=self.downsample_method,
                                auto_downsample=self.auto_downsample)


    def create_plot(self, plugin, container):
        self.source = plugin.find_source(self.source_name)
        #self.y_axis.setLabel(self.source.label, unitPrefix=self.source.unit)
        # TODO FIXME
        self.y_axis.setLabel('TESTING', unitPrefix='Hz')
        container.data_range.add_source(self.source, self)

    def update_range(self, low, high):
        data = self.source.get_range(low, high)
        t = np.arange(len(data))/self.source.fs + low
        return partial(self.plot.setData, t, data)


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

                #item.enableAutoRange(False, False)
                #item.disableAutoRange()
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
