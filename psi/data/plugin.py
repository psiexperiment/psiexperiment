from atom.api import ContainerList, Typed
from enaml.workbench.api import Plugin

import numpy as np
import pandas as pd

from .trial_data import TrialData


TRIAL_POINT = 'psi.data.trial'
PHYSIOLOGY_POINT = 'psi.data.physiology'


class DataPlugin(Plugin):

    parameters = ContainerList()
    trial_log = Typed(pd.DataFrame)

    def prepare(self):
        pass

    def prepare_trial_data(self, context_info):
        #columns = context_info.keys()
        arrays = dict((k, np.array([], dtype=i['dtype'])) \
                       for k, i in context_info.items())
        self.trial_log = pd.DataFrame(arrays)
        point = self.workbench.get_extension_point(TRIAL_POINT)
        for extension in point.extensions:
            for data in extension.get_children(TrialData):
                data.prepare(context_info)
            if extension.factory is not None:
                for data in extension.factory(self.workbench):
                    data.prepare(context_info)

    def process_trial(self, results):
        self.trial_log = self.trial_log.append(results, ignore_index=True)
        point = self.workbench.get_extension_point(TRIAL_POINT)
        for extension in point.extensions:
            for data in extension.get_children(TrialData):
                data.process_trial(results)
                data.trial_log_updated(self.trial_log)
            if extension.factory is not None:
                for data in extension.factory():
                    data.process_trial(results)
                    data.trial_log_updated(self.trial_log)
