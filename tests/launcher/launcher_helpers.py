'''
Stand-ins shared by the launcher tests.
'''
from pathlib import Path
from types import SimpleNamespace

from atom.api import Int

from psi.launcher.process_manager import ProcessManager


class FakeParadigm:
    '''
    Just enough of a `ParadigmDescription` for `Experiment`.
    '''
    def __init__(self, name='tone', modes=None, plugins=(), preferences=()):
        self.name = name
        self.full_name = f'lab.paradigms.{name}'
        self.title = name.title()
        self.info = {} if modes is None else {'modes': modes}
        self.plugins = [
            SimpleNamespace(id=p, title=p, required=False, info={})
            for p in plugins
        ]
        self._preferences = [Path(p) for p in preferences]

    def list_preferences(self):
        return self._preferences


class FakeProcessManager(ProcessManager):
    '''
    A process manager that records what it is asked to run instead of
    starting a websocket server and subprocesses.
    '''
    #: Number of times the next queued command would have been run.
    opened = Int()

    def __init__(self):
        # Deliberately skips ProcessManager.__init__.
        import threading
        self.lock = threading.Lock()

    def open_next_subprocess(self):
        self.opened += 1
