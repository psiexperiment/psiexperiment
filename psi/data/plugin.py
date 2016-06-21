from atom.api import ContainerList, Typed
from enaml.workbench.api import Plugin

import numpy as np
import pandas as pd

from .sink import Sink


SINK_POINT = 'psi.data.sink'


class DataPlugin(Plugin):

    _sinks = Typed(list, [])

    inputs = Typed(dict, {})
    context_info = Typed(dict, {})
    trial_log = Typed(pd.DataFrame)
    event_log = Typed(pd.DataFrame)

    def start(self):
        self._refresh_sinks()
        self._bind_observers()

        # Listen to changes on the context items so that we can update the
        # trial log accordingly.
        context = self.workbench.get_plugin('psi.context')
        context.observe('context_items', self._context_items_changed)

    def stop(self):
        self._unbind_observers()

    def _refresh_sinks(self):
        sinks = []
        point = self.workbench.get_extension_point(SINK_POINT)
        for extension in point.extensions:
            sinks.extend(extension.get_children(Sink))
        self._sinks = sinks

    def _bind_observers(self):
        self.workbench.get_extension_point(SINK_POINT) \
            .observe('extensions', self._refresh_sinks)

    def _unbind_observers(self):
        self.workbench.get_extension_point(SINK_POINT) \
            .unobserve('extensions', self._refresh_sinks)

    def _context_items_changed(self, items=None):
        context = self.workbench.get_plugin('psi.context')
        self.context_info = context.get_context_info()
        for sink in self._sinks:
            sink.context_info_updated(self.context_info)

    def _prepare_trial_log(self):
        ci = self.context_info.items()
        arrays = dict((k, np.array([], dtype=i['dtype'])) for k, i in ci)
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
        self.inputs = controller._inputs.copy()
        for sink in self._sinks:
            sink.prepare(self)

    def finalize(self):
        for sink in self._sinks:
            sink.finalize()

    def process_trial(self, results):
        self.trial_log = self.trial_log.append(results, ignore_index=True)
        for sink in self._sinks:
            sink.trial_log_updated(self.trial_log)
            sink.process_trial(results)

    def process_event(self, event, timestamp):
        row = {'event': event, 'timestamp': timestamp}
        self.event_log = self.event_log.append(row, ignore_index=True)
        for sink in self._sinks:
            sink.event_log_updated(self.event_log)
            sink.process_event(event, timestamp)

    def process_ai(self, name, data):
        for sink in self._sinks:
            sink.process_ai(name, data)
