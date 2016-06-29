from atom.api import Unicode, Float, Tuple, Int
from enaml.core.api import Declarative, d_

from psi.core.chaco.api import ChannelDataRange, add_time_axis, add_default_grids
from chaco.api import LinearMapper, OverlayPlotContainer, DataRange1D


class PlotContainer(Declarative):

    title = d_(Unicode())
    label = d_(Unicode())


class TimeContainer(PlotContainer):

    trig_delay = d_(Float())
    span = d_(Float())

    def create_container(self, plugin):
        index_range = ChannelDataRange(trig_delay=self.trig_delay, span=self.span)
        index_mapper = LinearMapper(range=index_range)
        container = OverlayPlotContainer(padding=[20, 20, 50, 50])
        for child in self.children:
            plot = child.create_plot(plugin, index_mapper)
            container.add(plot)

        # Add the time axis to the final plot
        add_time_axis(plot)
        add_default_grids(plot, major_index=5, minor_index=1)
        return container


class Plot(Declarative):

    line_color = d_(Tuple())
    fill_color = d_(Tuple())


class ChannelPlot(Plot):

    source = d_(Unicode())
    value_range = d_(Tuple(Float(), Float()))

    def create_plot(self, plugin, index_mapper):
        from psi.core.chaco.api import ChannelPlot
        source = plugin.find_source(self.source)
        index_mapper.range.sources.append(source)
        value_range = DataRange1D(low_setting=self.value_range[0],
                                  high_setting=self.value_range[1])
        value_mapper = LinearMapper(range=value_range)
        return ChannelPlot(source=source, index_mapper=index_mapper,
                           value_mapper=value_mapper,
                           line_color=self.line_color)


class ExtremesChannelPlot(ChannelPlot):

    decimation_threshold = d_(Int(5))

    def create_plot(self, plugin, index_mapper):
        from psi.core.chaco.api import ExtremesChannelPlot
        source = plugin.find_source(self.source)
        index_mapper.range.sources.append(source)
        value_range = DataRange1D(low_setting=self.value_range[0],
                                  high_setting=self.value_range[1])
        value_mapper = LinearMapper(range=value_range)
        return ExtremesChannelPlot(source=source, index_mapper=index_mapper,
                                   value_mapper=value_mapper,
                                   line_color=self.line_color)


class TimeseriesPlot(Plot):

    source = d_(Unicode())
    rising_event = d_(Unicode())
    falling_event = d_(Unicode())
    rect_center = d_(Float())
    rect_height = d_(Float())

    def create_plot(self, plugin, index_mapper):
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
