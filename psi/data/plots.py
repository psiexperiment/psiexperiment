import logging
log = logging.getLogger(__name__)

from atom.api import Unicode, Float, Tuple, Int, Typed
from enaml.core.api import Declarative, d_

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


class FFTContainer(PlotContainer):

    freq_lb = d_(Float(0.1e3))
    freq_ub = d_(Float(100e3))

    def prepare(self, plugin):
        index_range = BaseChannelDataRange(low_setting=self.freq_lb,
                                           high_setting=self.freq_ub)
        index_mapper = LogMapper(range=index_range)
        for child in self.children:
            plot = child.create_plot(plugin, index_mapper)
            self.container.add(plot)

        add_default_grids(plot, major_index=5, minor_index=1)
        axis = PlotAxis(component=plot, orientation='bottom', title='Frequency (Hz)')
        plot.underlays.append(axis)


class Plot(Declarative):

    line_color = d_(Tuple())
    fill_color = d_(Tuple())


class ChannelPlot(Plot):

    source = d_(Unicode())
    value_range = d_(Tuple(Float(), Float()))
    axis_label = d_(Unicode())

    def create_plot(self, plugin, index_mapper):
        log.info('Creating channel plot for {}'.format(self.source))
        from psi.core.chaco.api import ChannelPlot
        source = plugin.find_source(self.source)
        index_mapper.range.sources.append(source)
        value_range = DataRange1D(low_setting=self.value_range[0],
                                  high_setting=self.value_range[1])
        value_mapper = LinearMapper(range=value_range)
        plot = ChannelPlot(source=source, index_mapper=index_mapper,
                           value_mapper=value_mapper,
                           line_color=self.line_color)
        if self.axis_label:
            axis = PlotAxis(component=plot, orientation='left',
                            title=self.axis_label)
            plot.underlays.append(axis)
        return plot


class ExtremesChannelPlot(ChannelPlot):

    decimation_threshold = d_(Int(5))

    def create_plot(self, plugin, index_mapper):
        log.info('Creating extremes channel plot for {}'.format(self.source))
        from psi.core.chaco.api import ExtremesChannelPlot
        source = plugin.find_source(self.source)
        index_mapper.range.sources.append(source)
        value_range = DataRange1D(low_setting=self.value_range[0],
                                  high_setting=self.value_range[1])
        value_mapper = LinearMapper(range=value_range)
        return ExtremesChannelPlot(source=source, 
                                   index_mapper=index_mapper,
                                   value_mapper=value_mapper,
                                   line_color=self.line_color)


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
