from atom.api import Typed
from enaml.core.api import Declarative


class Sink(Declarative):

    def process_trial(self, results):
        pass

    def process_event(self, event, timestamp):
        pass

    def process_ai(self, name, data):
        pass

    def trial_log_updated(self, trial_log):
        pass

    def event_log_updated(self, event_log):
        pass

    def context_info_updated(self, context_info):
        pass

    def prepare(self, plugin):
        pass

    def finalize(self):
        pass

    def set_current_time(self, timestamp):
        pass

    def get_source(self, source_name):
        raise AttributeError
