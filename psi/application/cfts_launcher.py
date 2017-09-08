import subprocess
import os.path
import datetime as dt

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
        dt_string = dt.datetime.now().strftime('%Y%m%d-%H%M')
        base_folder = self.base_folder.format(date_time=dt_string,
                                              experiment='ABR')
        subprocess.check_output(['psi', 'abr', base_folder])

    def _update_base_folder(self):
        data_root = get_config('DATA_ROOT')
        template = '{{date_time}} {experimenter} {animal} {ear} {note} {{experiment}}'
        formatted = template.format(experimenter=self.experimenter,
                                    animal=self.animal, ear=self.ear,
                                    note=self.note)
        self.base_folder = os.path.join(data_root, formatted)


def main():
    with enaml.imports():
        from psi.application.cfts_launcher_view import LauncherView
    app = QtApplication()
    launcher = Launcher()
    view = LauncherView(launcher=launcher)
    view.show()
    app.start()
