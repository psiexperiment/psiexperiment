import logging
log = logging.getLogger(__name__)

from functools import partial
import os.path
from pathlib import Path
import subprocess

from atom.api import Atom, Bool, Enum, List, Typed, Unicode
import enaml
from enaml.qt.qt_application import QtApplication

with enaml.imports():
    from psi.application.base_launcher_view import LauncherView

from psi import get_config
from psi.application import list_calibrations, list_io, list_preferences
from psi.application.experiment_description import get_experiments, ParadigmDescription


class SimpleLauncher(Atom):

    io = Typed(Path)
    experiment = Typed(ParadigmDescription)
    calibration = Typed(Path)
    preferences = Typed(Path)
    save_data = Bool(True)
    experimenter = Unicode()
    note = Unicode()

    experiment_type = Unicode()
    experiment_choices = List()

    root_folder = Typed(Path)
    base_folder = Typed(Path)
    template = '{{date_time}} {experimenter} {note} {experiment}'

    can_launch = Bool(False)

    available_io = List()
    available_calibrations = List()
    available_preferences = List()

    def _default_experiment(self):
        return self.experiment_choices[0]

    def _default_experiment_choices(self):
        return get_experiments(self.experiment_type)

    def _default_available_io(self):
        return list_io()

    def _update_choices(self):
        self._update_available_calibrations()
        self._update_available_preferences()


    def _update_available_calibrations(self):
        self.available_calibrations = list_calibrations(self.io)
        if not self.available_calibrations:
            self.calibration = None
            return

        if self.calibration not in self.available_calibrations:
            for calibration in self.available_calibrations:
                if calibration.stem == 'default':
                    self.calibration = calibration
                    break
            else:
                self.calibration = self.available_calibrations[0]

    def _update_available_preferences(self):
        if not self.experiment:
            return
        self.available_preferences = list_preferences(self.experiment)
        if not self.available_preferences:
            self.preferences = None
            return

        if self.preferences not in self.available_preferences:
            for preferences in self.available_preferences:
                if preferences.stem == 'default':
                    self.preferences = preferences
                    break
            else:
                self.preferences = self.available_preferences[0]

    def _default_io(self):
        return sorted(list_io())[0]

    def _default_root_folder(self):
        return get_config('DATA_ROOT')

    def _observe_io(self, event):
        self._update_choices()

    def _observe_save_data(self, event):
        self._update()

    def _observe_experiment(self, event):
        self._update_choices()
        self._update()

    def _observe_experimenter(self, event):
        self._update()

    def _observe_note(self, event):
        self._update()

    def _update(self):
        if not self.save_data and self.experiment:
            self.can_launch = True
            self.base_folder = None
            return

        exclude = ['preferences', 'save_data', 'base_folder', 'can_launch']
        vals = {m: getattr(self, m) for m in self.members() if m not in exclude}
        for k, v in vals.items():
            if k == 'note':
                continue
            if not v:
                self.can_launch = False
                self.base_folder = None
                return

        vals['experiment'] = vals['experiment'].name
        self.base_folder = self.root_folder / self.template.format(**vals)
        self.can_launch = True

    def launch_subprocess(self):
        args = ['psi', self.experiment.name]
        plugins = [p.name for p in self.experiment.plugins if p.selected]
        if self.save_data:
            args.append(str(self.base_folder))
        if self.preferences:
            args.extend(['--preferences', str(self.preferences)])
        if self.io:
            args.extend(['--io', str(self.io)])
        if self.calibration:
            args.extend(['--calibration', str(self.calibration)])
        for plugin in plugins:
            args.extend(['--plugins', plugin])
        log.info('Launching subprocess: %s', ' '.join(args))
        print(' '.join(args))
        subprocess.check_output(args)
        self._update_choices()


class AnimalLauncher(SimpleLauncher):

    animal = Unicode()

    template = '{{date_time}} {experimenter} {animal} {note} {experiment}'

    def _observe_animal(self, event):
        self._update()


class EarLauncher(AnimalLauncher):

    ear = Enum('right', 'left')

    template = '{{date_time}} {experimenter} {animal} {ear} {note} {experiment}'

    def _observe_ear(self, event):
        self._update()


def _launch(klass, experiment_type, root_folder=None):
    app = QtApplication()
    if root_folder is None:
        root_folder = get_config('DATA_ROOT')
    launcher = klass(root_folder=root_folder, experiment_type=experiment_type)
    view = LauncherView(launcher=launcher)
    view.show()
    app.start()


main_calibration = partial(_launch, SimpleLauncher, 'calibration',
                           get_config('CAL_ROOT'))
main_cohort = partial(_launch, SimpleLauncher, 'cohort')
main_animal = partial(_launch, AnimalLauncher, 'animal')
main_ear = partial(_launch, EarLauncher, 'ear')
