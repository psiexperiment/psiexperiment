from atom.api import Typed
from enaml.core.api import Declarative


class TrialData(Declarative):

    context_info = Typed(dict, {})

    def context_info_updated(self, context_info):
        pass

    def trial_log_updated(self, trial_log):
        pass

    def event_log_updated(self, event_log):
        pass

    def process_trial(self, results):
        pass

    def process_event(self, event, timestamp):
        pass

    def prepare(self):
        pass
