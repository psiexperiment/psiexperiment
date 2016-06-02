from atom.api import ContainerList, Typed
from enaml.workbench.api import Plugin

import numpy as np
import pandas as pd

from .trial_data import TrialData


TRIAL_POINT = 'psi.data.trial'


class DataPlugin(Plugin):

    context_info = Typed(dict, {})
    trial_log = Typed(pd.DataFrame)
    _trial_data = Typed(list, [])

    def start(self):
        self._refresh_trial_data()
        self._bind_observers()
        self._context_items_changed()

        # Listen to changes on the context items so that we can update the trial
        # log accordingly.
        context = self.workbench.get_plugin('psi.context')
        context.observe('context_items', self._context_items_changed)

    def stop(self):
        self._unbind_observers()

    def _refresh_trial_data(self):
        trial_data = []
        point = self.workbench.get_extension_point(TRIAL_POINT)
        for extension in point.extensions:
            trial_data.extend(extension.get_children(TrialData))
            if extension.factory is not None:
                trial_data.extend(extension.factory(self.workbench))
        self._trial_data = trial_data

    def _bind_observers(self):
        self.workbench.get_extension_point(TRIAL_POINT) \
            .observe('extensions', self._refresh_trial_data)

    def _unbind_observers(self):
        self.workbench.get_extension_point(TRIAL_POINT) \
            .unobserve('extensions', self._refresh_trial_data)

    def _context_items_changed(self, items=None):
        context = self.workbench.get_plugin('psi.context')
        self.context_info = context.get_context_info()
        for data in self._trial_data:
            data.context_info_updated(self.context_info.copy())

    def prepare_trial_data(self):
        arrays = dict((k, np.array([], dtype=i['dtype'])) \
                      for k, i in self.context_info.items())
        self.trial_log = pd.DataFrame(arrays)
        for data in self._trial_data:
            data.trial_log_updated(self.trial_log)
            data.prepare()

    def process_trial(self, results):
        self.trial_log = self.trial_log.append(results, ignore_index=True)
        for data in self._trial_data:
            data.trial_log_updated(self.trial_log)
            data.process_trial(results)
