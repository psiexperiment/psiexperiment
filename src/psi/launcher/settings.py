import logging
log = logging.getLogger(__name__)

from pathlib import Path

from atom.api import Atom, Enum, List, Str, Typed

from psi import get_config

from .process_manager import ProcessManager


class LauncherSettings(Atom):
    '''
    Base class for the state of a launcher that runs psi experiments.

    The widgets in `psi.launcher.widgets` (e.g., `ExperimentSequence`) expect
    their `settings` to provide `process_manager` and `run_experiment`, which
    this class supplies. A launcher subclasses it and implements
    `build_experiment`, which turns a (frozen) experiment into the `psi`
    command line and environment variables to run it with. Everything
    specific to the lab's setup (equipment, calibrations, how data files are
    named) lives in the subclass.

    Saving and loading the launcher configuration is left to the subclass.
    '''
    logging_level = Enum('trace', 'debug', 'info', 'warning', 'error')('info')
    animal = Str()
    experimenter = Str()
    note = Str()
    standard_notes = List(Str())
    standard_note = Str()

    #: Handles launching experiments and communicating with the running psi
    #: processes.
    process_manager = Typed(ProcessManager)

    def _default_process_manager(self):
        manager = ProcessManager()
        manager.subscribe(self.process_event)
        return manager

    def process_event(self, event, uid):
        '''
        Called for each event relayed by the process manager.

        Parameters
        ----------
        event : str
            One of `psi.launcher.process_manager.RELAYED_EVENTS` or
            'subprocess_exited'.
        uid : object
            Identifier passed when the command was queued.
        '''
        pass

    def freeze_experiment(self, experiment, mode):
        '''
        Return a frozen copy of `experiment` to run in `mode`.

        Override to fix additional settings on the frozen experiment (e.g., the
        ear selected in the GUI).
        '''
        return experiment.freeze(mode)

    def build_experiment(self, experiment, save=True):
        '''
        Build the command and environment for running an experiment.

        Parameters
        ----------
        experiment : FrozenExperiment
            Experiment to run, in the mode given by `experiment.mode`.
        save : bool
            If True, the experiment should save its data.

        Returns
        -------
        cmd : list of str
            Command line to run. See `psi_command` for a starting point.
        env : dict
            Environment variables to set for the subprocess. Must include
            `base_env()` so that the subprocess can report back to the
            process manager.
        '''
        raise NotImplementedError

    def base_env(self):
        '''
        Environment variables every experiment needs to talk to the launcher.
        '''
        return {'PSI_WEBSOCKETS_URI': self.process_manager.ws_server.connected_uri}

    def data_path(self, *parts):
        '''
        Path under `PSI_DATA_ROOT` to save an experiment to.

        The name is built from `parts` (empty parts are dropped) and starts
        with the `{date_time}` placeholder that psi fills in at the start of
        the experiment.
        '''
        name = ' '.join(str(p) for p in ('{date_time}', *parts) if p)
        name = ' '.join(name.split())
        return Path(get_config('PSI_DATA_ROOT')) / name

    def psi_command(self, experiment, data_path=None):
        '''
        Command line for running `experiment` with `psi`.

        Covers the settings every launcher shares: the paradigm, where to save
        the data (not saved if `data_path` is None), the preferences file, the
        logging level and the selected plugins. Extend the returned list with
        anything else the launcher needs.
        '''
        cmd = ['psi', experiment.paradigm.full_name]
        if data_path is not None:
            cmd.append(str(data_path))
        if experiment.preference:
            cmd.extend(['--preferences', experiment.preference])
        cmd.extend(['--debug-level-console', self.logging_level.upper()])
        for plugin in experiment.plugins:
            cmd.extend(['--plugin', plugin])
        return cmd

    def prepare_sequence(self, sequence, save=True):
        '''
        Queue a sequence of frozen experiments to run.

        Every command is built before any is queued so that a failure partway
        through doesn't leave a partial sequence queued. Anything left over
        from a previous run that was aborted is discarded.
        '''
        commands = [self.build_experiment(e, save=save) for e in sequence]
        self.process_manager.clear_commands()
        for cmd, env in commands:
            self.process_manager.add_command(cmd, env)

    def run_sequence(self, sequence, save=True, autostart=False):
        '''
        Run a sequence of frozen experiments, one after the other.

        With `autostart`, each experiment starts as soon as its window is
        ready and the next one opens when it ends. Otherwise the operator
        starts each experiment.
        '''
        self.prepare_sequence(sequence, save=save)
        self.process_manager.autostart = autostart
        self.process_manager.open_next_subprocess()

    def run_experiment(self, experiment, mode, save=True, autostart=False):
        '''
        Run a single experiment in `mode`. Called by the mode buttons of
        `ExperimentSequence`.
        '''
        frozen = self.freeze_experiment(experiment, mode)
        self.run_sequence([frozen], save=save, autostart=autostart)
