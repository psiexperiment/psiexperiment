import logging
log = logging.getLogger(__name__)

from pathlib import Path

from atom.api import ContainerList, Typed, Unicode
from enaml.application import deferred_call
from enaml.workbench.api import Plugin

import numpy as np
import pandas as pd

from psi import get_config
from psi.core.enaml.api import load_manifests

from .sink import Sink
from .plots import BasePlotContainer, MultiPlotContainer


SINK_POINT = 'psi.data.sinks'
PLOT_POINT = 'psi.data.plots'


class DataPlugin(Plugin):

    _sinks = Typed(list, [])
    _containers = Typed(list, [])

    inputs = Typed(dict, {})
    context_info = Typed(dict, {})

    def start(self):
        self._refresh_sinks()
        self._refresh_plots()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _refresh_sinks(self, event=None):
        log.debug('Refreshing sinks')
        sinks = []
        point = self.workbench.get_extension_point(SINK_POINT)
        for extension in point.extensions:
            sinks.extend(extension.get_children(Sink))
        load_manifests(sinks, self.workbench)
        self._sinks = sinks

    def _refresh_plots(self, event=None):
        containers = []
        point = self.workbench.get_extension_point(PLOT_POINT)
        for extension in point.extensions:
            containers.extend(extension.get_children(BasePlotContainer))
            containers.extend(extension.get_children(MultiPlotContainer))
        load_manifests(containers, self.workbench)
        log.debug('Found %d plot containers', len(containers))
        self._containers = containers

    def _bind_observers(self):
        self.workbench.get_extension_point(SINK_POINT) \
            .observe('extensions', self._refresh_sinks)
        self.workbench.get_extension_point(PLOT_POINT) \
            .observe('extensions', self._refresh_plots)

    def _unbind_observers(self):
        self.workbench.get_extension_point(SINK_POINT) \
            .unobserve('extensions', self._refresh_sinks)
        self.workbench.get_extension_point(PLOT_POINT) \
            .unobserve('extensions', self._refresh_plots)

    def set_base_path(self, base_path):
        for sink in self._sinks:
            sink.set_base_path(base_path)

    def find_plot_container(self, plot_container_name):
        for container in self._containers:
            if container.name == plot_container_name:
                return containers
        m = 'Plot container {} not available'
        raise AttributeError(m.format(plot_container_name))

    def find_viewbox(self, viewbox_name):
        for container in self._containers:
            for viewbox in container.children:
                if viewbox.name == viewbox_name:
                    return viewbox
        m = 'Viewbox {} not available'
        raise AttributeError(m.format(viewbox_name))

    def find_plot(self, plot_name):
        for container in self._containers:
            for viewbox in container.children:
                for plot in viewbox.children:
                    if plot.name == plot_name:
                        return plot
        m = 'Plot {} not available'
        raise AttributeError(m.format(plot_name))

    def find_sink(self, sink_name):
        for sink in self._sinks:
            if sink.name == sink_name:
                return sink
        raise AttributeError(f'Sink {sink_name} not available')
