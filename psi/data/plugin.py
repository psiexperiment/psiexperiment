from atom.api import ContainerList, Typed
from enaml.workbench.api import Plugin

import numpy as np
import pandas as pd

from .trial_data import TrialData


TRIAL_POINT = 'psi.data.trial'


class DataPlugin(Plugin):

    context_info = Typed(dict, {})
    _data = Typed(list, [])

    trial_log = Typed(pd.DataFrame)
    event_log = Typed(pd.DataFrame)

    def start(self):
        self._refresh_data()
        self._bind_observers()
        self._context_items_changed()

        # Listen to changes on the context items so that we can update the trial
        # log accordingly.
        context = self.workbench.get_plugin('psi.context')
        context.observe('context_items', self._context_items_changed)

    def stop(self):
        self._unbind_observers()

    def _refresh_data(self):
        data = []
        point = self.workbench.get_extension_point(TRIAL_POINT)
        for extension in point.extensions:
            data.extend(extension.get_children(TrialData))
            if extension.factory is not None:
                data.extend(extension.factory(self.workbench))
        self._data = data

    def _bind_observers(self):
        self.workbench.get_extension_point(TRIAL_POINT) \
            .observe('extensions', self._refresh_data)

    def _unbind_observers(self):
        self.workbench.get_extension_point(TRIAL_POINT) \
            .unobserve('extensions', self._refresh_data)

    def _context_items_changed(self, items=None):
        context = self.workbench.get_plugin('psi.context')
        self.context_info = context.get_context_info()
        for data in self._data:
            data.context_info_updated(self.context_info.copy())

    def _prepare_trial_log(self):
        ci = self.context_info.items()
        arrays = dict((k, np.array([], dtype=i['dtype'])) for k, i in ci)
        self.trial_log = pd.DataFrame(arrays)
        for data in self._data:
            data.trial_log_updated(self.trial_log)

    def _prepare_event_log(self):
        arrays = dict([
            ('timestamp', np.array([], dtype=np.dtype('float32'))), 
            ('event', np.array([], dtype=np.dtype('S512'))), 
        ])
        self.event_log = pd.DataFrame(arrays)
        for data in self._data:
            data.event_log_updated(self.event_log)

    def prepare_trial_data(self):
        self._prepare_trial_log()
        self._prepare_event_log()
        for data in self._data:
            data.prepare()

    def process_trial(self, results):
        self.trial_log = self.trial_log.append(results, ignore_index=True)
        for data in self._data:
            data.trial_log_updated(self.trial_log)
            data.process_trial(results)

    def process_event(self, event, timestamp):
        row = {'event': event, 'timestamp': timestamp}
        self.event_log = self.event_log.append(row, ignore_index=True)
        for data in self._data:
            data.event_log_updated(self.event_log)
            data.process_event(event, timestamp)

    #def prepare_channels(self, channels):
    #    for data in self._channel_data:
    #        data.prepare_channels(channels)
