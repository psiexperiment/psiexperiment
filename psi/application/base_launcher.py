import os.path
import subprocess

from atom.api import Atom, Unicode, Bool, Enum
import enaml
from enaml.qt.qt_application import QtApplication

from psi import get_config


with enaml.imports():
    from psi.application.base_launcher_view import LauncherView


class SimpleLauncher(Atom):

    io = Unicode()
    experiment = Unicode()
    settings = Unicode()
    save_data = Bool(True)
    experimenter = Unicode()
    note = Unicode()

    root_folder = Unicode()
    base_folder = Unicode()
    template = '{{date_time}} {experimenter} {note} {experiment}'

    can_launch = Bool(False)

    def _default_io(self):
        return get_config('SYSTEM')

    def _default_root_folder(self):
        return get_config('DATA_ROOT')

    def _observe_save_data(self, event):
        self._update()

    def _observe_experiment(self, event):
        self._update()

    def _observe_experimenter(self, event):
        self._update()

    def _observe_note(self, event):
        self._update()

    def _update(self):
        if not self.save_data and self.experiment:
            self.can_launch = True
            self.base_folder = ''
            return

        exclude = ['settings', 'save_data', 'base_folder', 'can_launch']
        vals = {m: getattr(self, m) for m in self.members() if m not in exclude}
        for k, v in vals.items():
            if k == 'note':
                continue
            if not v:
                self.can_launch = False
                self.base_folder = ''
                return

        formatted = self.template.format(**vals)
        self.base_folder = os.path.join(self.root_folder, formatted)
        self.can_launch = True

    def launch_subprocess(self):
        args = ['psi', self.experiment]
        if self.save_data:
            args.append(self.base_folder)
        if self.settings:
            args.extend(['--preferences', self.settings])
        if self.io:
            args.extend(['--io', self.io])
        subprocess.check_output(args)


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


def main_calibration():
    experiments =['speaker_calibration', 'pistonphone_calibration',
                  'pt_calibration_golay', 'pt_calibration_chirp']

    app = QtApplication()
    launcher = SimpleLauncher(root_folder=get_config('CAL_ROOT'))
    view = LauncherView(launcher=launcher, experiments=experiments)
    view.show()
    app.start()


def main_cohort():
    experiments = ['noise_exposure']
    app = QtApplication()
    launcher = SimpleLauncher()
    view = LauncherView(launcher=launcher, experiments=experiments)
    view.show()
    app.start()


def main_animal():
    experiments = ['appetitive_gonogo_food']
    app = QtApplication()
    launcher = AnimalLauncher()
    view = LauncherView(launcher=launcher, experiments=experiments)
    view.show()
    app.start()


def main_ear():
    experiments = ['speaker_calibration', 'abr', 'abr_with_eeg_view']
    app = QtApplication()
    launcher = EarLauncher()
    view = LauncherView(launcher=launcher, experiments=experiments)
    view.show()
    app.start()
    app.stop()


if __name__ == '__main__':
    main_calibration()
