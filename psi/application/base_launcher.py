import logging
log = logging.getLogger(__name__)

from functools import partial
import datetime as dt
import os.path
from pathlib import Path
import subprocess

from atom.api import Atom, Bool, Enum, List, Typed, Str
import enaml
from enaml.qt.qt_application import QtApplication

with enaml.imports():
    from enaml.stdlib.message_box import critical
    from psi.application.base_launcher_view import LauncherView

from psi import get_config
from psi.util import get_tagged_values
from psi.application import (get_default_io, list_calibrations, list_io,
                             list_preferences)
from psi.application.experiment_description import (get_experiments,
                                                    ParadigmDescription)


class SimpleLauncher(Atom):

    io = Typed(Path)
    experiment = Typed(ParadigmDescription).tag(template=True, required=True)
    calibration = Typed(Path)
    preferences = Typed(Path)
    save_data = Bool(True)
    experimenter = Str().tag(template=True, required=True)
    note = Str().tag(template=True)

    experiment_type = Str()
    experiment_choices = List()

    root_folder = Typed(Path)
    base_folder = Typed(Path).tag(required=False)
    wildcard = Str()
    template = '{{date_time}} {experimenter} {note} {experiment}'
    wildcard_template = '*{experiment}'
    use_prior_preferences = Bool(False)

    can_launch = Bool(False).tag(required=False)

    available_io = List()
    available_calibrations = List()
    available_preferences = List().tag(required=False)

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
        return get_default_io()

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
        exclude = list(get_tagged_values(self, 'required', False).keys())
        exclude_save = ['experimenter', 'animal', 'ear']
        if not self.save_data:
            exclude.extend(exclude_save)

        template_vals = get_tagged_values(self, 'template')
        required_vals = get_tagged_values(self, 'required')
        for k, v in required_vals.items():
            if k == 'note':
                continue
            if k == 'save_data':
                continue
            if not v:
                self.can_launch = False
                self.base_folder = None
                return

        if self.save_data:
            vals['experiment'] = vals['experiment'].name
            self.base_folder = self.root_folder / self.template.format(**vals)

        template_vals['experiment'] = template_vals['experiment'].name
        self.base_folder = self.root_folder / self.template.format(**template_vals)
        self.wildcard = self.wildcard_template.format(**template_vals)
        self.can_launch = True

    def get_preferences(self):
        if not self.use_prior_preferences:
            return self.preferences
        options = []
        for match in self.root_folder.glob(self.wildcard):
            if (match / 'final.preferences').exists():
                n = match.name.split(' ')[0]
                date = dt.datetime.strptime(n, '%Y%m%d-%H%M%S')
                options.append((date, match / 'final.preferences'))
            elif (match / 'initial.preferences').exists():
                n = match.name.split(' ')[0]
                date = dt.datetime.strptime(n, '%Y%m%d-%H%M%S')
                options.append((date, match / 'initial.preferences'))
        options.sort(reverse=True)
        if len(options):
            return options[0][1]
        m = f'Could not find prior preferences for {self.experiment_type}'
        raise ValueError(m)

    def launch_subprocess(self):
        args = ['psi', self.experiment.name]
        plugins = [p.name for p in self.experiment.plugins if p.selected]
        if self.save_data:
            args.append(str(self.base_folder))
        if self.preferences:
            args.extend(['--preferences', str(self.get_preferences())])
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

    animal = Str().tag(template=True, required=True)

    template = '{{date_time}} {experimenter} {animal} {note} {experiment}'
    wildcard_template = '*{animal}*{experiment}'

    def _observe_animal(self, event):
        self._update()


class EarLauncher(AnimalLauncher):

    ear = Enum('right', 'left').tag(template=True)

    template = '{{date_time}} {experimenter} {animal} {ear} {note} {experiment}'
    wildcard_template = '*{animal} {ear}*{experiment}'

    def _observe_ear(self, event):
        self._update()


def launch(klass, experiment_type, root_folder='DATA_ROOT', view_klass=None):
    app = QtApplication()
    try:
        if root_folder.endswith('_ROOT'):
            root_folder = get_config(root_folder)
        if view_klass is None:
            view_klass = LauncherView
        launcher = klass(root_folder=root_folder, experiment_type=experiment_type)
        view = view_klass(launcher=launcher)
        view.show()
        app.start()
        return True
    except Exception as e:
        mesg = f'Unable to load configuration data.\n\n{e}'
        critical(None, 'Software not configured', mesg)
        raise


main_calibration = partial(launch, SimpleLauncher, 'calibration', 'CAL_ROOT')
main_cohort = partial(launch, SimpleLauncher, 'cohort')
main_animal = partial(launch, AnimalLauncher, 'animal')
main_ear = partial(launch, EarLauncher, 'ear')
