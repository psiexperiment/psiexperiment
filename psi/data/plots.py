import logging
log = logging.getLogger(__name__)

from functools import partial
import struct

import numpy as np
import pyqtgraph as pg

from atom.api import (Unicode, Float, Tuple, Int, Typed, Property, Atom, Bool)
from enaml.core.api import Declarative, d_
from enaml.application import deferred_call, timed_call

from psi.core.enaml.api import PSIContribution


################################################################################
# Supporting classes
################################################################################
class ChannelDataRange(object):

    def __init__(self, container, span):
        self.container = container
        self.span = span
        self.current_time = 0
        self.current_range = None

    def add_source(self, source, plot):
        source.observe('added', partial(self.source_added, plot=plot))

    def source_added(self, event, plot):
        self.current_time = max(self.current_time, event['value']['ub'])
        spans = self.current_time // self.span
        high_value = (spans+1)*self.span
        low_value = high_value-self.span
        if self.current_range != (low_value, high_value):
            self.container.update_range(low_value, high_value)
        else:
            plot.update_range(low_value, high_value)

    def update_range(self, range):
        self.current_range = (low_value, high_value)
        self.container.update_range(low_value, high_value)


class CustomViewBox(pg.ViewBox):

    def wheelEvent(self, ev, axis=None):
        mask = np.array(self.state['mouseEnabled'], dtype=np.float)
        if axis is not None and axis >= 0 and axis < len(mask):
            mv = mask[axis]
            mask[:] = 0
            mask[axis] = mv
        s = ((mask*0.02)+1)**(ev.delta()*self.state['wheelScaleFactor'])
        #center = Point(fn.invertQTransform(self.childGroup.transform()).map(ev.pos()))

        self._resetTarget()
        self.scaleBy(s, center=None)
        self.sigRangeChangedManually.emit(self.state['mouseEnabled'])
        ev.accept()


################################################################################
# Containers (defines a shared set of containers across axes)
################################################################################
class PlotContainer(PSIContribution):

    title = d_(Unicode())
    label = d_(Unicode())
    container = Typed(pg.GraphicsWidget)

    def _default_container(self):
        return pg.GraphicsLayout()


class Plot(PSIContribution):

    line_color = d_(Tuple())
    fill_color = d_(Tuple())



class TimeContainer(PlotContainer):

    span = d_(Float())
    plot_item = Typed(pg.PlotItem)
    data_range = Typed(ChannelDataRange)
    plots = Typed(list, [])

    # TODO: not implemented
    trig_delay = d_(Float())
    major_grid_index = d_(Float(5))
    minor_grid_index = d_(Float(1))

    y_min = d_(Float(-1))
    y_max = d_(Float(1))

    def prepare(self, plugin):
        viewbox = CustomViewBox()
        self.plot_item = pg.PlotItem(viewBox=viewbox)
        self.plot_item.enableAutoRange(False, False)
        self.plot_item.setXRange(0, self.span)
        self.plot_item.setYRange(self.y_min, self.y_max)
        self.container.addItem(self.plot_item, 0, 0)
        self.data_range = ChannelDataRange(self, self.span)
        for child in self.children:
            plot = child.create_plot(plugin, self)
            self.plots.append(plot)

    def update_range(self, low, high):
        for child in self.children:
            child.update_range(low, high)


################################################################################
# Plots
################################################################################
class Plot(PSIContribution):

    line_color = d_(Tuple())
    fill_color = d_(Tuple())


class ChannelPlot(Plot):

    source_name = d_(Unicode())
    value_range = d_(Tuple(Float(), Float()))

    source = Typed(object)
    plot = Typed(object)

    def create_plot(self, plugin, container):
        self.source = plugin.find_source(self.source_name)
        container.data_range.add_source(self.source, self)
        self.plot = container.plot_item.plot()
        return self.plot

    def update_range(self, low, high):
        data = self.source.get_range(low, high)
        t = np.arange(len(data))/self.source.fs
        self.plot.setData(t, data)


class ExtremesChannelPlot(ChannelPlot):

    container = Typed(object)
    downsample = Int()
    time = Typed(object)

    y_min = Typed(object)
    y_max = Typed(object)
    fill = Typed(object)

    def decimate_extremes(self, y, downsample):
        # If data is empty, return imediately
        if y.size == 0:
            return np.array([]), np.array([])

        # Determine the fragment size that we are unable to decimate.  A
        # downsampling factor of 5 means that we perform the operation in chunks
        # of 5 samples.  If we have only 13 samples of data, then we cannot
        # decimate the last 3 samples and will simply discard them.
        offset = int(y.shape[-1] % downsample)
        y = y[..., :-offset].reshape((-1, downsample))
        return y.min(-1), y.max(-1)

    def create_plot(self, plugin, container):
        self.container = container

        pen = pg.mkPen('k', width=1)
        self.plot = pg.PlotCurveItem(pen=pen)
        self.container.plot_item.addItem(self.plot)

        self.source = plugin.find_source(self.source_name)
        self.container.data_range.add_source(self.source, self)
        self.container.plot_item.setMouseEnabled(x=True, y=True)

        # TODO: For some reason this crashes if we attempt to connect directly
        # to the compute pixel size function.
        def handle(*args, obj=self, **kwargs):
            self.compute_pixel_size()

        self.container.plot_item.vb.geometryChanged.connect(handle)
        return self.plot

    def update_buffers(self, *args, **kwargs):
        data_samples = int(self.container.span/self.source.fs)
        decimated_samples = int(data_samples/self.downsample)
        self._data_limits = 0, 0
        self._data_buffer = np.empty(x_range, dtype=np.float)
        self._decimated_buffer = np.empty((2, x_range), dtype=np.float)

    def compute_pixel_size(self, *args, **kwargs):
        try:
            pixel_width, _ = self.container.plot_item.vb.viewPixelSize()
            self.downsample = int(pixel_width*self.source.fs)
            time = np.arange(self.container.span*self.source.fs)/self.source.fs
            self.time = time[::self.downsample]
        except:
            pass

    def update_range(self, low, high):
        if self.downsample == 0:
            return

        #cache_low, cache_high = self._data_limits
        #if cache_low > low:

        #lb = int((low-cache_low)*self.source.fs)
        #ub = int((high-cache_low)*self.source.fs)

        #samples = self._data_buffer[lb:ub]
        #if 

        #if cache_low > low:


        log.trace('Downsampling signal at {}'.format(self.downsample))
        data = self.source.get_range(low, high)
        y_min, y_max = self.decimate_extremes(data, self.downsample)
        n = len(y_min)
        t = np.arange(len(y_min))/self.source.fs*self.downsample
        x = np.column_stack([t, t]).ravel()
        y = np.column_stack([y_min, y_max]).ravel()
        deferred_call(self.plot.setData, x, y)


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
