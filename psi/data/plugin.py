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
from .plots import BasePlot, BasePlotContainer, ViewBox

import textwrap


SINK_POINT = 'psi.data.sinks'
PLOT_POINT = 'psi.data.plots'


class DataPlugin(PSIPlugin):

    _sinks = Typed(dict, {})
    _containers = Typed(dict, {})
    _plots = Typed(dict, {})
    _viewboxes = Typed(dict, {})

    # TODO, is this still needed
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
        plugin_info = {
            BasePlotContainer: 'name',
            BasePlot: 'name',
            ViewBox: 'name',
        }
        plugins = self.load_multiple_plugins(PLOT_POINT, plugin_info)

        # These represent only those found at the top level of an Enaml
        # hierarchy (i.e., the level right below the extension). If the ViewBox
        # already belongs to a container or the BasePlot already belongs to a
        # ViewBox, we will not see it here. Hence, the reason we use
        # `find_viewbox` to find the plots we need.
        self._containers = plugins[BasePlotContainer]
        self._plots = plugins[BasePlot]
        self._viewboxes = plugins[ViewBox]

        for plot in self._plots.values():
            if not plot.viewbox_name:
                raise ValueError(f'Must specify viewbox to embed {plot} plot in.')
            plot.set_parent(self.find_viewbox(plot.viewbox_name))

        self.load_manifests(self._containers.values())
        # Have containers update their viewbox layouts
        for c in self._containers.values():
            c._update_container()

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
        '''
        Set base_path for all sinks

        Parameters
        ----------
        base_path : instance of pathlib.Path
            Base path where data should be saved. Sinks will create files
            inside this folder.
        is_temp : bool
            If True, data will be discarded at end of experiment. Some sinks
            may alter their behavior based on this (i.e., not save data).

        This is used to notify sinks of where data should be saved once
        experiment begins. We don't initialize base_path until the user presses
        the "start" button since the filename may contain the timestamp the
        experiment was started.
        '''
        self.base_path = Path(base_path)
        for sink in self._sinks.values():
            sink.set_base_path(base_path, is_temp)

    def find_plot_container(self, plot_container_name):
        available_names = []
        for container in self._containers:
            if container.name == plot_container_name:
                return container
            available_names.append(container.name)

        available_names = ', '.join(available_names)
        m = f'Plot container {plot_container_name} not available. ' \
            f'Valid choices are {available_names}'
        raise AttributeError(m)

    def find_viewbox(self, viewbox_name):
        available_names = []
        for container in self._containers.values():
            for viewbox in container.children:
                if viewbox.name == viewbox_name:
                    return viewbox
                available_names.append(viewbox.name)

        available_names = ', '.join(available_names)
        m = f'Viewbox {viewbox_name} not available. Valid choices are {available_names}.'
        raise AttributeError(m)

    def find_plot(self, plot_name):
        for container in self._containers.values():
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

    def find_source(self, source_name):
        for sink in self._sinks.values():
            if hasattr(sink, 'get_source'):
                try:
                    return sink.get_source(source_name)
                except:
                    continue
        raise AttributeError(f'Could not find source "{source_name}"')
