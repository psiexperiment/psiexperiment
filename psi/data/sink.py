from atom.api import Typed
from enaml.core.api import Declarative
from psi.core.enaml.api import PSIContribution


class Sink(PSIContribution):

    def process_trials(self, results):
        pass

    def process_event(self, event, timestamp):
        pass

    def process_ai_continuous(self, name, data):
        pass

    def process_ai_epochs(self, name, data):
        pass

    def trial_log_updated(self, trial_log):
        pass

    def event_log_updated(self, event_log):
        pass

    def context_info_updated(self, context_info):
        pass

    def prepare(self, plugin):
        pass

    def finalize(self, workbench):
        pass

    def set_current_time(self, name, timestamp):
        pass

    def set_base_path(self, base_path):
        pass

    def get_source(self, source_name):
        raise AttributeError
