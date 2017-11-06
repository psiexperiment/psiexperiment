import subprocess
import os.path
import datetime as dt

from atom.api import Atom, Unicode

import enaml
from enaml.qt.qt_application import QtApplication

from psi import get_config


class Launcher(Atom):

    experiment = Unicode()
    preferences = Unicode()
    experimenter = Unicode()
    animal = Unicode()
    note = Unicode()
    base_folder = Unicode()

    def _observe_experimenter(self, event):
        self._update_base_folder()

    def _observe_animal(self, event):
        self._update_base_folder()

    def _observe_note(self, event):
        self._update_base_folder()

    def start_experiment(self):
        dt_string = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        base_folder = self.base_folder.format(date_time=dt_string)
        cmd = ['psi', self.experiment, base_folder]
        if self.preferences:
            cmd.extend(['--preferences', self.preferences])
        subprocess.check_output(cmd)

    def _update_base_folder(self):
        data_root = get_config('DATA_ROOT')
        template = '{{date_time}} {experimenter} {animal} {note} {experiment}'
        formatted = template.format(experimenter=self.experimenter,
                                    animal=self.animal, note=self.note,
                                    experiment=self.experiment)
        self.base_folder = os.path.join(data_root, formatted)


def main():
    with enaml.imports():
        from psi.application.gui_launcher_view import LauncherView
    app = QtApplication()
    launcher = Launcher()
    view = LauncherView(launcher=launcher)
    view.show()
    app.start()


if __name__ == '__main__':
    main()
