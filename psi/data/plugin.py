import os.path
import datetime as dt
import logging
log = logging.getLogger(__name__)

from atom.api import ContainerList, Typed, Unicode
from enaml.workbench.api import Plugin

import numpy as np
import pandas as pd

from psi import get_config

from .sink import Sink
from .plots import PlotContainer


SINK_POINT = 'psi.data.sinks'
PLOT_POINT = 'psi.data.plots'


class DataPlugin(Plugin):

    _sinks = Typed(list, [])
    _plots = Typed(list, [])
    _containers = Typed(dict, {})

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
            sink.load_manifest(self.workbench)
        self._sinks = sinks

    def _refresh_plots(self, event=None):
        plots = []
        point = self.workbench.get_extension_point(PLOT_POINT)
        for extension in point.extensions:
            plots.extend(extension.get_children(PlotContainer))
        for container in plots:
            container.load_manifest(self.workbench)
            for child in container.children:
                child.load_manifest(self.workbench)
        self._plots = plots

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

    def _prepare_trial_log(self):
        ci = self.context_info.items()
        arrays = dict((k, np.array([], dtype=i.dtype)) for k, i in ci)
        self.trial_log = pd.DataFrame(arrays)

    def _prepare_event_log(self):
        arrays = dict([
            ('timestamp', np.array([], dtype=np.dtype('float32'))),
            ('event', np.array([], dtype=np.dtype('S512'))),
        ])
        self.event_log = pd.DataFrame(arrays)

    def prepare(self):
        self._prepare_trial_log()
        self._prepare_event_log()
        controller = self.workbench.get_plugin('psi.controller')
        # TODO: somehow make results available if needed even if not persisted
        # forever?
        self.inputs = {k: v for k, v in controller._inputs.items() if v.save}
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

    def process_ai_continuous(self, name, data, **kw):
        for sink in self._sinks:
            sink.process_ai_continuous(name, data)

    def process_ai_epochs(self, name, data, **kw):
        for sink in self._sinks:
            sink.process_ai_epochs(name, data)

    def set_current_time(self, name, timestamp):
        for sink in self._sinks:
            sink.set_current_time(name, timestamp)

    def set_base_path(self, base_path):
        self.base_path = base_path
        for sink in self._sinks:
            sink.set_base_path(self.base_path)

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
