import logging
log = logging.getLogger(__name__)

from functools import partial
import struct

import numpy as np

from atom.api import Unicode, Float, Tuple, Int, Typed, Property, Atom
from enaml.core.api import Declarative, d_
from enaml.application import deferred_call

from psi.core.chaco.api import ChannelDataRange, add_time_axis, add_default_grids
from chaco.api import LinearMapper, LogMapper, OverlayPlotContainer, DataRange1D, PlotAxis
from psi.core.chaco.base_channel_data_range import BaseChannelDataRange

# TODO: refactor so overlays and underlays can also be declarative

from psi.core.enaml.api import PSIContribution


class PlotContainer(PSIContribution):

    manifest = 'psi.data.plots_manifest.PlotContainerManifest'

    title = d_(Unicode())
    label = d_(Unicode())
    container = Typed(object)

    def _default_container(self):
        return OverlayPlotContainer(padding=[50, 50, 50, 50])


class TimeContainer(PlotContainer):

    trig_delay = d_(Float())
    span = d_(Float())
    major_grid_index = d_(Float(5))
    minor_grid_index = d_(Float(1))

    def prepare(self, plugin):
        index_range = ChannelDataRange(trig_delay=self.trig_delay,
                                       span=self.span)
        index_mapper = LinearMapper(range=index_range)
        for child in self.children:
            plot = child.create_plot(plugin, index_mapper)
            self.container.add(plot)
        add_time_axis(plot)
        add_default_grids(plot, major_index=self.major_grid_index,
                          minor_index=self.minor_grid_index)


class Plot(PSIContribution):

    line_color = d_(Tuple())
    fill_color = d_(Tuple())


class TimeseriesPlot(Plot):

    source = d_(Unicode())
    rising_event = d_(Unicode())
    falling_event = d_(Unicode())
    rect_center = d_(Float())
    rect_height = d_(Float())

    def create_plot(self, plugin, index_mapper):
        m = 'Creating timeseries plot for {} with events {} and {}'
        log.info(m.format(self.source, self.rising_event, self.falling_event))
        from psi.core.chaco.api import TimeseriesPlot
        source = plugin.find_source(self.source)
        index_mapper.range.sources.append(source)
        value_range = DataRange1D(low_setting=0, high_setting=1)
        value_mapper = LinearMapper(range=value_range)
        return TimeseriesPlot(source=source,
                              index_mapper=index_mapper,
                              value_mapper=value_mapper,
                              rect_center=self.rect_center,
                              rect_height=self.rect_height,
                              rising_event=self.rising_event,
                              falling_event=self.falling_event,
                              fill_color=self.fill_color,
                              line_color=self.line_color)


class FFTChannelPlot(Plot):

    source = d_(Unicode())
    value_range = d_(Tuple(Float(), Float()))
    time_span = d_(Float(1))
    axis_label = d_(Unicode())
    reference = d_(Float(1))

    def create_plot(self, plugin, index_mapper):
        from psi.core.chaco.api import FFTChannelPlot
        source = plugin.find_source(self.source)
        index_mapper.range.sources.append(source)
        value_range = DataRange1D(low_setting=self.value_range[0],
                                  high_setting=self.value_range[1])
        value_mapper = LinearMapper(range=value_range)
        plot = FFTChannelPlot(source=source,
                              reference=self.reference,
                              time_span=self.time_span,
                              index_mapper=index_mapper,
                              value_mapper=value_mapper,
                              line_color=self.line_color)
        if self.axis_label:
            axis = PlotAxis(component=plot, orientation='left',
                            title=self.axis_label)
            plot.underlays.append(axis)
        return plot


import pyqtgraph as pg

class PGPlotContainer(PlotContainer):

    manifest = __name__ + '_manifest.PGPlotContainerManifest'

    title = d_(Unicode())
    label = d_(Unicode())

    container = Typed(pg.GraphicsWidget)

    def _default_container(self):
        return pg.GraphicsLayout()


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


class PGTimeContainer(PGPlotContainer):

    span = d_(Float())
    plot_item = Typed(pg.PlotItem)
    data_range = Typed(ChannelDataRange)

    plots = Typed(list, [])

    def prepare(self, plugin):
        viewbox = CustomViewBox()
        self.plot_item = pg.PlotItem(viewBox=viewbox)
        self.plot_item.enableAutoRange(False, False)
        self.plot_item.setXRange(0, self.span)
        self.plot_item.setYRange(-1e-3, 1e-3)
        self.container.addItem(self.plot_item, 0, 0)
        self.data_range = ChannelDataRange(self, self.span)
        for child in self.children:
            plot = child.create_plot(plugin, self)
            self.plots.append(plot)

    def update_range(self, low, high):
        for child in self.children:
            child.update_range(low, high)


class PGPlot(PSIContribution):

    line_color = d_(Tuple())
    fill_color = d_(Tuple())


class PGChannelPlot(PGPlot):

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


def arrayToQPath(x, y_min, y_max):
    """Convert an array of x,y coordinats to QPainterPath as efficiently as poss
ible.
    The *connect* argument may be 'all', indicating that each point should be
    connected to the next; 'pairs', indicating that each pair of points
    should be connected, or an array of int32 values (0 or 1) indicating
    connections.
    """
    path = pg.QtGui.QPainterPath()
    n = x.shape[0]*2

    # create empty array, pad with extra space on either end
    arr = np.empty(n+2, dtype=[('x', '>f8'), ('y', '>f8'), ('c', '>i4')])
    # write first two integers
    byteview = arr.view(dtype=np.ubyte)
    byteview[:12] = 0
    byteview.data[12:20] = struct.pack('>ii', n, 0)
    # Fill array with vertex values
    arr[1:-1]['x'][::2] = x
    arr[1:-1]['x'][1::2] = x
    arr[1:-1]['y'][::2] = y_min
    arr[1:-1]['y'][1::2] = y_max

    # decide which points are connected by lines
    arr[1:-1]['c'][::2] = 1
    arr[1:-1]['c'][1::2] = 0

    # write last 0
    lastInd = 20*(n+1)
    byteview.data[lastInd:lastInd+4] = struct.pack('>i', 0)
    # create datastream object and stream into path

    ## Avoiding this method because QByteArray(str) leaks memory in PySide
    #buf = QtCore.QByteArray(arr.data[12:lastInd+4])  # I think one unnecessary copy happens here

    path.strn = byteview.data[12:lastInd+4] # make sure data doesn't run away
    try:
        buf = pg.QtCore.QByteArray.fromRawData(path.strn)
    except TypeError:
        buf = pg.QtCore.QByteArray(bytes(path.strn))
    ds = pg.QtCore.QDataStream(buf)
    ds >> path
    return path


class MultiLine(pg.QtGui.QGraphicsPathItem):

    def __init__(self):
        self.path = pg.arrayToQPath(np.array([]), np.array([]), 'pairs')
        super().__init__(self.path)
        self.setPath(self.path)

    def setData(self, x, y_min, y_max):
        self.path = arrayToQPath(x, y_min, y_max)
        self.setPath(self.path)
        self.prepareGeometryChange()
        self.update()


class PGExtremesChannelPlot(PGChannelPlot):

    container = Typed(object)
    downsample = Int()
    time = Typed(object)

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
        self.plot = MultiLine()
        self.source = plugin.find_source(self.source_name)
        self.container.data_range.add_source(self.source, self)
        self.container.plot_item.addItem(self.plot)
        self.container.plot_item.setMouseEnabled(x=False, y=True)
        self.container.plot_item.vb.geometryChanged.connect(self.compute_pixel_size)
        return self.plot

    def compute_pixel_size(self):
        pixel_width, _ = self.container.plot_item.vb.viewPixelSize()
        self.downsample = int(pixel_width*self.source.fs)
        time = np.arange(self.container.span*self.source.fs)/self.source.fs
        self.time = time[::self.downsample]

    def update_range(self, low, high):
        if self.downsample != 0:
            data = self.source.get_range(low, high)
            y_min, y_max = self.decimate_extremes(data, self.downsample)
            n = len(y_min)
            t = self.time[:n]
            deferred_call(self.plot.setData, t, y_min, y_max)


class CustomViewBox(pg.ViewBox):

    def wheelEvent(self, ev, axis=None):
        mask = np.array(self.state['mouseEnabled'], dtype=np.float)
        if axis is not None and axis >= 0 and axis < len(mask):
            mv = mask[axis]
            mask[:] = 0
            mask[axis] = mv
        s = ((mask * 0.02) + 1) ** (ev.delta() * self.state['wheelScaleFactor']) # actual scaling factor

        #center = Point(fn.invertQTransform(self.childGroup.transform()).map(ev.pos()))

        self._resetTarget()
        self.scaleBy(s, center=None)
        self.sigRangeChangedManually.emit(self.state['mouseEnabled'])
        ev.accept()


class PGEpochAverageGridContainer(PGPlotContainer):

    manifest = __name__ + '_manifest.PGEpochAverageGridContainerManifest'

    items = d_(Typed(list))
    source_name = d_(Unicode())
    source = Typed(object)

    trial_log = Typed(object)

    cumulative_average = Typed(dict, {})
    cumulative_var = Typed(dict, {})
    cumulative_n = Typed(dict, {})

    grid = Typed(tuple)
    plots = Typed(dict)
    time = Typed(object)

    def _default_grid(self):
        return set(), set()

    def prepare(self, plugin):
        self.source = plugin.find_source(self.source_name)
        self.source.observe('added', self.epochs_acquired)
        self.cumulative_average = {}
        self.cumulative_var = {}
        self.cumulative_n = {}
        self.items = [
            {'name': 'target_tone_level'},
            {'name': 'target_tone_frequency'},
        ]

    def prepare_grid(self, iterable):
        self.items = [
            {'name': 'target_tone_level'},
            {'name': 'target_tone_frequency'},
        ]
        keys = [self.extract_key(c) for c in iterable]
        self.update_grid(keys)

    def extract_key(self, context):
        return tuple(context[i['name']] for i in self.items)

    def epochs_acquired(self, event):
        for epoch in event['value']:
            self.process_epoch(epoch)
        deferred_call(self.update_plots)

    def process_epoch(self, data):
        key = self.extract_key(data['metadata'])
        epoch = data['epoch']

        n = self.cumulative_n.get(key, 0) + 1
        time, average = self.cumulative_average.get(key, (None, epoch))
        delta = epoch-average
        average = average + delta/n
        delta2 = epoch-average
        M2 = delta*delta2
        var = M2/(n-1)

        # TODO: Would be nice to move this out (e.g., precompute)?
        if time is None:
            time = np.arange(len(average))/self.source.fs

        self.cumulative_average[key] = time, average
        self.cumulative_var[key] = var
        self.cumulative_n[key] = n

    def update_grid(self, keys):
        rows, cols = map(set, zip(*keys))
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
                if not (row, col) in keys:
                    continue
                viewbox = CustomViewBox()
                item = pg.PlotItem(viewBox=viewbox)
                self.container.addItem(item, r+1, c+1)
                if base_item is None:
                    base_item = item
                else:
                    #item.setXLink(base_item)
                    #item.setYLink(base_item)
                    pass
                item.enableAutoRange(False, False)
                item.setMouseEnabled(x=False, y=True)
                item.setXRange(0, 8.5e-3)
                #curve = pg.PlotCurveItem()
                #item.addItem(curve)
                plots[row, col] = item.plot()
        self.plots = plots
        self.grid = (rows, cols)

    def update_plots(self):
        keys = self.cumulative_average.keys()
        self.update_grid(keys)
        for k, (time, average) in self.cumulative_average.items():
            self.plots[k].setData(time, average)
