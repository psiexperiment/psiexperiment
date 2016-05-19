from atom.api import Int, Typed, Unicode
import numpy as np

from .base_plugin import BaseController


class AppetitivePlugin(BaseController):

    trial = Int(0)
    consecutive_nogo = Int(0)
    rng = Typed(np.random.RandomState)
    trial_type = Unicode()

    def next_selector(self):
        try:
            max_nogo = self.context.get_value('max_nogo')
            go_probability = self.context.get_value('go_probability')
            score = self.context.get_value('score')
        except KeyError:
            self.trial_type = 'go_remind'
            return 'remind'

        if self._remind_requested:
            self.trial_type = 'go_remind'
            return 'remind'
        elif self.consecutive_nogo >= max_nogo:
            self.trial_type = 'go_forced'
            return 'go'
        elif score == 'FA':
            self.trial_type = 'nogo_repeat'
            return 'nogo'
        else:
            if self.rng.uniform() <= go_probability:
                self.trial_type = 'go'
                return 'go'
            else:
                self.trial_type = 'nogo'
                return 'nogo'

    def start_experiment(self):
        try:
            self.rng = np.random.RandomState()
            self.context.apply_changes()
            #self.core.invoke_command('psiexperiment.data.prepare')
            selector = self.next_selector()
            self.context.next_setting(selector, save_prior=False)
            self.state = 'running'
            self.start_trial()
        except Exception as e:
            # TODO - provide user interface
            raise

    def start_trial(self):
        # Hack to force the context to be computed. Usually we would do a lot
        # more here otherwise.
        self.trial += 1
        self.context.get_values()

    def end_trial(self, response):
        if self.trial_type in ('nogo', 'nogo_repeat'):
            self.consecutive_nogo += 1
            if response == 'spout':
                score = 'FA'
            elif response == 'poke':
                score = 'CR'
        else:
            self.consecutive_nogo = 0
            if response == 'spout':
                score = ' HIT'
            elif response == 'poke':
                score = 'MISS'

        results = {
            'response': response,
            'trial_type': self.trial_type,
            'score': score,
            'correct': score in ('CR', 'HIT'),
        }
        self.context.set_values(results)
        self.core.invoke_command('psiexperiment.data.process_trial')

        # Apply pending changes that way any parameters (such as repeat_FA or
        # go_probability) are reflected in determining the next trial type.
        if self._apply_requested:
            self.context.apply_changes()
            self._apply_requested = False
        selector = self.next_selector()
        self.context.next_setting(selector, save_prior=True)

        if self._pause_requested:
            self.state = 'paused'
            self._pause_requested = False
        else:
            self.start_trial()
