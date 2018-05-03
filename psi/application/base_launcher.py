import os.path

from atom.api import Atom, Unicode, Bool, Enum
import enaml
from enaml.qt.qt_application import QtApplication

from psi import get_config


with enaml.imports():
    from psi.application.base_launcher_view import LauncherView


class AnimalLauncher(Atom):

    experiment = Unicode()
    settings = Unicode()
    save_data = Bool(True)
    experimenter = Unicode()
    animal = Unicode()
    note = Unicode()
    base_folder = Unicode()

    def _observe_save_data(self, event):
        self._update()

    def _observe_experiment(self, event):
        self._update()

    def _observe_experimenter(self, event):
        self._update()

    def _observe_animal(self, event):
        self._update()

    def _observe_note(self, event):
        self._update()

    def _update(self):
        data_root = get_config('DATA_ROOT')
        fmt = '{{date_time}} {experimenter} {animal} {note} {experiment}'
        formatted = fmt.format(experiment=self.experiment,
                               experimenter=self.experimenter,
                               animal=self.animal,
                               note=self.note)
        self.base_folder = os.path.join(data_root, formatted)


class EarLauncher(AnimalLauncher):

    ear = Enum('right', 'left')

    def _observe_ear(self, event):
        self._update()

    def _update(self):
        data_root = get_config('DATA_ROOT')
        fmt = '{{date_time}} {experimenter} {animal} {ear} {note} {experiment}'
        formatted = fmt.format(experiment=self.experiment,
                               experimenter=self.experimenter,
                               animal=self.animal,
                               ear=self.ear,
                               note=self.note)
        self.base_folder = os.path.join(data_root, formatted)


def main_animal():
    experiments = ['appetitive_gonogo_food']
    app = QtApplication()
    launcher = AnimalLauncher()
    view = LauncherView(launcher=launcher, experiments=experiments)
    view.show()
    app.start()


def main_ear():
    experiments = ['calibration', 'abr']
    app = QtApplication()
    launcher = EarLauncher()
    view = LauncherView(launcher=launcher, experiments=experiments)
    view.show()
    app.start()
    app.stop()
