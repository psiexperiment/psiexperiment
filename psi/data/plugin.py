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
from .plots import PlotContainer


SINK_POINT = 'psi.data.sinks'
PLOT_POINT = 'psi.data.plots'


class DataPlugin(Plugin):

    _sinks = Typed(list, [])
    _containers = Typed(list, [])

    inputs = Typed(dict, {})
    context_info = Typed(dict, {})
    trial_log = Typed(pd.DataFrame)
    event_log = Typed(pd.DataFrame)

    base_path = Unicode()

    def start(self):
        self._refresh_sinks()
        self._refresh_plots()
        self._bind_observers()

        # Listen to changes on the context items so that we can update the
        # trial log accordingly.
        # TODO: Observing context_items seems a bit hackish since this is not
        # the shadow copy but the live copy. What do we do?
        context = self.workbench.get_plugin('psi.context')
        context.observe('context_items', self._context_items_changed)

    def stop(self):
        self._unbind_observers()

    def _refresh_sinks(self, event=None):
        sinks = []
        point = self.workbench.get_extension_point(SINK_POINT)
        for extension in point.extensions:
            sinks.extend(extension.get_children(Sink))
        for sink in sinks:
            deferred_call(sink.load_manifest, self.workbench)
        self._sinks = sinks

    def _refresh_plots(self, event=None):
        containers = []
        point = self.workbench.get_extension_point(PLOT_POINT)
        for extension in point.extensions:
            containers.extend(extension.get_children(PlotContainer))
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

    def _default_trial_log(self):
        ci = self.context_info.items()
        arrays = dict((k, np.array([], dtype=i.dtype)) for k, i in ci)
        return pd.DataFrame(arrays)

    def _default_event_log(self):
        arrays = dict([
            ('timestamp', np.array([], dtype=np.dtype('float32'))),
            ('event', np.array([], dtype=np.dtype('S512'))),
        ])
        return pd.DataFrame(arrays)

    def prepare(self):
        for sink in self._sinks:
            sink.prepare(self)

    def finalize(self):
        for sink in self._sinks:
            sink.finalize(self.workbench)

    def process_trials(self, results):
        self.trial_log = self.trial_log.append(results, ignore_index=True)
        for sink in self._sinks:
            sink.trial_log_updated(self.trial_log)
            sink.process_trials(results)

    def process_event(self, event, timestamp):
        row = {'event': event, 'timestamp': timestamp}
        self.event_log = self.event_log.append(row, ignore_index=True)
        for sink in self._sinks:
            sink.event_log_updated(self.event_log)
            sink.process_event(event, timestamp)

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
