from atom.api import Atom, Bool, Enum, List, Unicode, Typed


class PluginDescription(Atom):

    name = Unicode()
    title = Unicode()
    required = Bool()
    selected = Bool()
    manifest = Unicode()


class ParadigmDescription(Atom):

    name = Unicode()
    title = Unicode()
    plugins = List()
    type = Enum('ear', 'animal', 'cohort', 'calibration')

    def enable_plugin(self, plugin_name):
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                plugin.selected = True
                return
        valid_plugins = ', '.join(p.name for p in self.plugins)
        raise ValueError(f'Plugin {plugin_name} does not exist. ' \
                         f'Valid options are {valid_plugins}')


class ExperimentDescription(Atom):

    name = Unicode()
    title = Unicode()
    io_manifest = Unicode()
    paradigm = Typed(ParadigmDescription)


abr_controller = PluginDescription(
    name='controller',
    title='Controller',
    required=True,
    manifest='psi.application.experiment.abr_base.ControllerManifest',
)


speaker_calibration_controller = PluginDescription(
    name='controller',
    title='Controller',
    required=True,
    manifest='psi.application.experiment.speaker_calibration.ControllerManifest',
)


temperature_mixin = PluginDescription(
    name='temperature',
    title='Temperature display',
    required=False,
    manifest='psi.application.experiment.cfts_mixins.TemperatureMixinManifest',
)


eeg_view_mixin = PluginDescription(
    name='eeg_view',
    title='EEG display',
    required=False,
    manifest='psi.application.experiment.cfts_mixins.EEGViewMixinManifest',
)


in_ear_calibration_mixin = PluginDescription(
    name='in_ear_calibration',
    title='In-ear calibration',
    required=False,
    selected=True,
    manifest='psi.application.experiment.cfts_mixins.InEarCalibrationMixinManifest',
)


abr_experiment = ParadigmDescription(
    name='abr',
    title='ABR',
    type='ear',
    plugins=[
        abr_controller,
        temperature_mixin,
        eeg_view_mixin,
        in_ear_calibration_mixin,
    ]
)


speaker_calibration_experiment = ParadigmDescription(
    name='speaker_calibration',
    title='Speaker calibration',
    type='ear',
    plugins=[
        speaker_calibration_controller,
    ]
)


experiments = {
    'abr': abr_experiment,
    'speaker_calibration': speaker_calibration_experiment,
}


def get_experiments(type):
    return [e for e in experiments.values() if e.type == type]
