import logging
log = logging.getLogger(__name__)

import copy

from atom.api import Atom, Bool, Enum, List, Str, Typed

from psi.core.enaml.api import load_manifest


experiments = {}


################################################################################
# Core classes and utility functions
################################################################################
class PluginDescription(Atom):

    name = Str()
    title = Str()
    required = Bool()
    selected = Bool()
    manifest = Str()

    def __init__(self, manifest, selected=False, **kwargs):
        kwargs['manifest'] = manifest
        manifest = load_manifest(manifest)()
        for attr in ('name', 'title', 'required'):
            if attr not in kwargs:
                kwargs[attr] = getattr(manifest, attr)
        super().__init__(**kwargs)

    def copy(self, **kwargs):
        other = copy.copy(self)
        for k, v in kwargs.items():
            setattr(other, k, v)
        return other


class ParadigmDescription(Atom):

    name = Str()
    title = Str()
    plugins = List()
    type = Enum('ear', 'animal', 'cohort', 'calibration')

    def __init__(self, name, title, experiment_type, plugin_info):
        for pi in plugin_info:
            print(pi)
        plugins = [PluginDescription(*pi) for pi in plugin_info]
        super().__init__(name=name, title=title, type=experiment_type,
                         plugins=plugins)
        global experiments
        experiments[name] = self


    def enable_plugin(self, plugin_name):
        for plugin in self.plugins:
            if plugin.name == plugin_name:
                plugin.selected = True
                return
        valid_plugins = ', '.join(p.name for p in self.plugins)
        raise ValueError(f'Plugin {plugin_name} does not exist. ' \
                         f'Valid options are {valid_plugins}')

    def copy(self, **kwargs):
        other = copy.copy(self)
        for k, v in kwargs.items():
            setattr(other, k, v)
        return other


def get_experiments(type):
    return [e for e in experiments.values() if e.type == type]


################################################################################
# CFTS stuff
################################################################################
ParadigmDescription(
    'abr_io_editable', 'Configurable ABR (input-output)', 'ear', [
        ('psi.application.experiment.abr_io.ABRIOManifest',),
        ('psi.application.experiment.cfts_mixins.TemperatureMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.EEGViewMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.ABRInEarCalibrationMixinManifest', True),
    ]
)


ParadigmDescription(
    'abr_io', 'ABR (input-output)', 'ear', [
        ('psi.application.experiment.abr_io.ABRIOManifest',),
        ('psi.application.experiment.cfts_mixins.TemperatureMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.EEGViewMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.ABRInEarCalibrationMixinManifest', True),
    ]
)


ParadigmDescription(
    'dpoae_io', 'DPOAE input-output', 'ear', [
        ('psi.application.experiment.dpoae_io.ControllerManifest',),
        ('psi.application.experiment.cfts_mixins.TemperatureMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.DPOAEInEarCalibrationMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.MicrophoneSignalViewMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.MicrophoneFFTViewMixinManifest', True),
    ]
)


ParadigmDescription(
    'dpoae_ttl', 'DPOAE (TTL output)', 'ear', [
        ('psi.application.experiment.dpoae_time.ControllerManifest',),
        ('psi.application.experiment.cfts_mixins.TemperatureMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.DPOAEInEarCalibrationMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.MicrophoneSignalViewMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.MicrophoneFFTViewMixinManifest', True),
    ]
)


ParadigmDescription(
    'dpoae_contra', 'DPOAE (contra noise)', 'ear', [
        ('psi.application.experiment.dpoae_time.ControllerManifest',),
        ('psi.application.experiment.cfts_mixins.TemperatureMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.MicrophoneElicitorFFTViewMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.DPOAETimeNoiseMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.DPOAEInEarCalibrationMixinManifest', True),
        ('psi.application.experiment.cfts_mixins.DPOAEInEarNoiseCalibrationMixinManifest', True),
    ]
)

#################################################################################
## Calibration
#################################################################################
ParadigmDescription(
    'speaker_calibration_golay', 'Speaker calibration (Golay)', 'calibration', [
        ('psi.application.experiment.speaker_calibration.BaseSpeakerCalibrationManifest',),
        ('psi.application.experiment.calibration_mixins.GolayMixin',),
        ('psi.application.experiment.calibration_mixins.ToneValidateMixin',),
    ]
)


ParadigmDescription(
    'speaker_calibration_chirp', 'Speaker calibration (chirp)', 'calibration', [
        ('psi.application.experiment.speaker_calibration.BaseSpeakerCalibrationManifest',),
        ('psi.application.experiment.calibration_mixins.ChirpMixin',),
        ('psi.application.experiment.calibration_mixins.ToneValidateMixin',),
    ]
)


ParadigmDescription(
    'speaker_calibration_tone', 'Speaker calibration (tone)', 'calibration', [
        ('psi.application.experiment.speaker_calibration.BaseSpeakerCalibrationManifest',),
        ('psi.application.experiment.calibration_mixins.ToneMixin',),
    ]
)


ParadigmDescription(
    'pistonphone_calibration', 'Pistonphone calibration', 'calibration', [
        ('psi.application.experiment.pistonphone_calibration.PistonphoneCalibrationManifest',),
    ]
)

ParadigmDescription(
    'pt_calibration_chirp', 'Probe tube calibration (chirp)', 'calibration', [
        ('psi.application.experiment.pt_calibration.ChirpControllerManifest',),
    ],
)


ParadigmDescription(
    'pt_calibration_golay', 'Probe tube calibration (golay)', 'calibration', [
        ('psi.application.experiment.pt_calibration.GolayControllerManifest',),
    ],
)


#################################################################################
## Noise exposure
#################################################################################
ParadigmDescription(
    'noise_exposure', 'Noise exposure', 'cohort', [
        ('psi.application.experiment.noise_exposure.ControllerManifest',),
    ],
)


#################################################################################
## Behavior
#################################################################################
ParadigmDescription(
    'appetitive_gonogo_food', 'Appetitive GO-NOGO food', 'animal', [
        ('psi.application.experiment.appetitive.ControllerManifest',),
        ('psi.application.experiment.behavior_base.PelletDispenserMixinManifest', True),
    ],
)
