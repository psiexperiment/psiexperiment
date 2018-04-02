from psi.core.enaml.api import PSIContribution


class Sink(PSIContribution):

    def prepare(self, plugin):
        pass

    def finalize(self, plugin):
        pass

    def process_trials(self, results):
        pass

    def process_event(self, event, timestamp):
        pass

    def create_table(self, name, dataframe):
        pass

    def trial_log_updated(self, trial_log):
        pass

    def event_log_updated(self, event_log):
        pass

    def context_info_updated(self, context_info):
        pass

    def set_base_path(self, base_path):
        pass

    def get_source(self, source_name):
        raise AttributeError
