from atom.api import Enum, Bool, Typed
from enaml.workbench.plugin import Plugin


class BaseController(Plugin):
    # Tracks the state of the controller.
    state = Enum('initialized', 'running', 'paused', 'stopped')

    # Provides direct access to plugins rather than going through the core
    # command system. Right now the context plugin is so fundamentally important
    # to the controller that it would be cumbersome to use the core command
    # system.
    core = Typed(Plugin)
    context = Typed(Plugin)

    # We should not respond to changes during the course of a trial. These flags
    # indicate changes or requests from the user are pending and should be
    # processed when the opportunity arisese (e.g., at the end of the trial).
    _apply_requested = Bool(False)
    _remind_requested = Bool(False)
    _pause_requested = Bool(False)

    def start(self):
        self.core = self.workbench.get_plugin('enaml.workbench.core')
        self.context = self.workbench.get_plugin('psiexperiment.context')
        self.core.invoke_command('psiexperiment.data.prepare')

    def request_apply(self):
        self._apply_requested = True

    def request_remind(self):
        self._remind_requested = True

    def request_pause(self):
        self._pause_requested = True

    def request_resume(self):
        self.state == 'running'
        self.start_trial()

    def start_experiment(self):
        raise NotImplementedError

    def stop_experiment(self):
        self.state = 'stopped'

    def start_trial(self):
        raise NotImplementedError

    def end_trial(self):
        raise NotImplementedError
