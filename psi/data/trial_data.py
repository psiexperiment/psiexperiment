from enaml.core.api import Declarative


class TrialData(Declarative):

    def prepare(self, parameters, trial_log):
        pass

    def process_trial(self, trial_log):
        pass

    def trial_log_updated(self, trial_log):
        pass
