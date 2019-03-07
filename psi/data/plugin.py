import os.path
import datetime as dt
import logging
log = logging.getLogger(__name__)

from atom.api import ContainerList, Typed, Unicode
from enaml.application import deferred_call
from enaml.workbench.api import Plugin

import numpy as np
import pandas as pd

from psi import get_config
from psi.core.enaml.api import load_manifests

from .sink import Sink
from .plots import MultiPlotContainer, PlotContainer


SINK_POINT = 'psi.data.sinks'
PLOT_POINT = 'psi.data.plots'


class DataPlugin(Plugin):

    _sinks = Typed(list, [])
    _containers = Typed(list, [])

    inputs = Typed(dict, {})
    context_info = Typed(dict, {})

    base_path = Unicode()

    def start(self):
        self._refresh_sinks()
        self._refresh_plots()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _refresh_sinks(self, event=None):
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
            containers.extend(extension.get_children(PlotContainer))
            containers.extend(extension.get_children(MultiPlotContainer))
        load_manifests(containers, self.workbench)
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

    def _context_items_changed(self, items=None):
        context = self.workbench.get_plugin('psi.context')
        self.context_info = context.context_items.copy()
        for sink in self._sinks:
            sink.context_info_updated(self.context_info)

    def prepare(self):
        for sink in self._sinks:
            sink.prepare(self)

    def finalize(self):
        for sink in self._sinks:
            sink.finalize(self.workbench)

    def set_base_path(self, base_path):
        self.base_path = base_path
        for sink in self._sinks:
            sink.set_base_path(self.base_path)

    def get_base_path(self):
        return self.base_path

    def find_source(self, source_name):
        '''
        Find the source by quering the sinks in order until one of them returns
        the channel.
        '''
        for sink in self._sinks:
            try:
                return sink.get_source(source_name)
            except AttributeError:
                pass
        raise AttributeError('Source {} not available'.format(source_name))

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
