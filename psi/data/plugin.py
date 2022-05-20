import logging
log = logging.getLogger(__name__)

from pathlib import Path

from atom.api import ContainerList, Typed, Str
from enaml.application import deferred_call
from enaml.workbench.api import Plugin

import numpy as np
import pandas as pd

from psi import get_config
from psi.core.enaml.api import load_manifests, PSIPlugin

from .sink import Sink
from .plots import BasePlotContainer

import textwrap


SINK_POINT = 'psi.data.sinks'
PLOT_POINT = 'psi.data.plots'


class DataPlugin(PSIPlugin):

    _sinks = Typed(dict, {})
    _containers = Typed(list, [])

    inputs = Typed(dict, {})
    context_info = Typed(dict, {})

    base_path = Typed(Path)

    def start(self):
        self._refresh_sinks()
        self._refresh_plots()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _refresh_sinks(self, event=None):
        log.debug('Refreshing sinks')
        sinks = {}
        point = self.workbench.get_extension_point(SINK_POINT)
        for extension in point.extensions:
            for sink in extension.get_children(Sink):
                if sink.name in sinks:
                    self.raise_duplicate_error(sink, 'name', extension)
                sinks[sink.name] = sink
        load_manifests(sinks.values(), self.workbench)
        self._sinks = sinks

    def _refresh_plots(self, event=None):
        containers = []
        point = self.workbench.get_extension_point(PLOT_POINT)
        log.debug('Found %d extensions for %s', len(point.extensions),
                  PLOT_POINT)
        for extension in point.extensions:
            containers.extend(extension.get_children(BasePlotContainer))
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

    def set_base_path(self, base_path, is_temp):
        self.base_path = Path(base_path)
        for sink in self._sinks.values():
            sink.set_base_path(base_path, is_temp)

    def find_plot_container(self, plot_container_name):
        for container in self._containers:
            if container.name == plot_container_name:
                return container
        m = 'Plot container {} not available'
        raise AttributeError(m.format(plot_container_name))

    def find_viewbox(self, viewbox_name):
        available_names = []
        for container in self._containers:
            for viewbox in container.children:
                if viewbox.name == viewbox_name:
                    return viewbox
                available_names.append(viewbox.name)

        available_names = ', '.join(available_names)
        m = f'Viewbox {viewbox_name} not available. Valid choices are {available_names}.'
        raise AttributeError(m)

    def find_plot(self, plot_name):
        for container in self._containers:
            for viewbox in container.children:
                for plot in viewbox.children:
                    if plot.name == plot_name:
                        return plot
        m = 'Plot {} not available'
        raise AttributeError(m.format(plot_name))

    def find_sink(self, sink_name):
        try:
            return self._sinks[sink_name]
        except KeyError:
            raise AttributeError(f'Sink "{sink_name}" not available')
