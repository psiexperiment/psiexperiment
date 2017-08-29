import subprocess
import os.path

from atom.api import Atom, Unicode, Typed

import enaml
from enaml.qt.qt_application import QtApplication

from psi import get_config
from psi.application import add_default_options, launch_experiment



class Launcher(Atom):

    experimenter = Unicode()
    animal = Unicode()
    ear =  Unicode()
    note = Unicode()
    base_folder = Unicode()

    def _observe_experimenter(self, event):
        self._update_base_folder()

    def _observe_animal(self, event):
        self._update_base_folder()

    def _observe_ear(self, event):
        self._update_base_folder()

    def _observe_note(self, event):
        self._update_base_folder()

    def launch_abr(self):
        subprocess.check_output(['psi', 'abr', self.base_folder])

    def _update_base_folder(self):
        template = '{experimenter} {animal} {ear} {{experiment}}'
        self.base_folder = os.path.join(
            get_config('DATA_ROOT'),
            self.note,
            self.experimenter,
            self.animal,
            self.ear
        )


def main():
    with enaml.imports():
        from psi.application.cfts_launcher_view import LauncherView
    app = QtApplication()
    launcher = Launcher()
    view = LauncherView(launcher=launcher)
    view.show()
    app.start()
