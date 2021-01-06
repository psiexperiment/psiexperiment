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

        # Default values are loaded directly from the PluginManifest (but can
        # be overridden by kwargs).
        for attr in ('name', 'title', 'required'):
            kwargs.setdefault(attr, getattr(manifest, attr))
        super().__init__(**kwargs)


class ParadigmDescription(Atom):

    name = Str()
    title = Str()
    plugins = List()
    type = Enum('ear', 'animal', 'cohort', 'calibration')

    def __init__(self, name, title, experiment_type, plugin_info):
        plugins = [PluginDescription(*pi) for pi in plugin_info]
        super().__init__(name=name, title=title, type=experiment_type,
                         plugins=plugins)
        global experiments
        experiments[name] = self

    def enable_plugin(self, plugin_name):
        for p in self.plugins:
            if p.name == plugin_name:
                p.selected = True


def get_experiments(type):
    return [e for e in experiments.values() if e.type == type]


PATH = 'psi.application.experiment.'


################################################################################
# CFTS stuff
################################################################################
ParadigmDescription(
    # This is a much more flexible, more powerful ABR experiment interface.
    'abr_io_editable', 'Configurable ABR (input-output)', 'ear', [
        (PATH + 'abr_io.ABRIOManifest',),
        (PATH + 'cfts_mixins.TemperatureMixinManifest', True),
        (PATH + 'cfts_mixins.EEGViewMixinManifest', True),
        (PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest', True),
    ]
)


ParadigmDescription(
    # This is the default, simple ABR experiment that most users will want.  
    'abr_io', 'ABR (input-output)', 'ear', [
        (PATH + 'abr_io_simple.ABRIOSimpleManifest',),
        (PATH + 'cfts_mixins.TemperatureMixinManifest', True),
        (PATH + 'cfts_mixins.EEGViewMixinManifest', True),
        (PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest', True),
    ]
)


ParadigmDescription(
    'dpoae_io', 'DPOAE input-output', 'ear', [
        (PATH + 'dpoae_io.ControllerManifest',),
        (PATH + 'cfts_mixins.TemperatureMixinManifest', True),
        (PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest', True),
        (PATH + 'cfts_mixins.MicrophoneSignalViewMixinManifest', True),
        (PATH + 'cfts_mixins.MicrophoneFFTViewMixinManifest', True),
    ]
)


ParadigmDescription(
    'dpoae_ttl', 'DPOAE (TTL output)', 'ear', [
        (PATH + 'dpoae_time.ControllerManifest',),
        (PATH + 'cfts_mixins.TemperatureMixinManifest', True),
        (PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest', True),
        (PATH + 'cfts_mixins.MicrophoneSignalViewMixinManifest', True),
        (PATH + 'cfts_mixins.MicrophoneFFTViewMixinManifest', True),
    ]
)


ParadigmDescription(
    'dpoae_contra', 'DPOAE (contra noise)', 'ear', [
        (PATH + 'dpoae_time.ControllerManifest',),
        (PATH + 'cfts_mixins.TemperatureMixinManifest', True),
        (PATH + 'cfts_mixins.MicrophoneElicitorFFTViewMixinManifest', True),
        (PATH + 'cfts_mixins.DPOAETimeNoiseMixinManifest', True),
        (PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest', True),
        (PATH + 'cfts_mixins.DPOAEInEarNoiseCalibrationMixinManifest', True),
    ],
)


ParadigmDescription(
    'efr', 'EFR (SAM)', 'ear', [
        (PATH + 'efr.EFRManifest',),
        (PATH + 'cfts_mixins.MicrophoneSignalViewMixinManifest', True),
        (PATH + 'cfts_mixins.MicrophoneFFTViewMixinManifest', True),
    ],
)


ParadigmDescription(
    'speaker_calibration_chirp', 'Speaker calibration (chirp)', 'ear', [
        (PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',),
        (PATH + 'calibration_mixins.ChirpMixin',),
        (PATH + 'calibration_mixins.ToneValidateMixin',),
    ]
)


#################################################################################
## Calibration
#################################################################################
ParadigmDescription(
    'speaker_calibration_golay', 'Speaker calibration (Golay)', 'calibration', [
        (PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',),
        (PATH + 'calibration_mixins.GolayMixin',),
        (PATH + 'calibration_mixins.ToneValidateMixin',),
    ]
)


ParadigmDescription(
    'speaker_calibration_chirp', 'Speaker calibration (chirp)', 'calibration', [
        (PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',),
        (PATH + 'calibration_mixins.ChirpMixin',),
        (PATH + 'calibration_mixins.ToneValidateMixin',),
    ]
)


ParadigmDescription(
    'speaker_calibration_tone', 'Speaker calibration (tone)', 'calibration', [
        (PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',),
        (PATH + 'calibration_mixins.ToneMixin',),
    ]
)


ParadigmDescription(
    'pistonphone_calibration', 'Pistonphone calibration', 'calibration', [
        (PATH + 'pistonphone_calibration.PistonphoneCalibrationManifest',),
    ]
)


ParadigmDescription(
    'pt_calibration_chirp', 'Probe tube calibration (chirp)', 'calibration', [
        (PATH + 'pt_calibration.ChirpControllerManifest',),
    ],
)


ParadigmDescription(
    'pt_calibration_golay', 'Probe tube calibration (golay)', 'calibration', [
        (PATH + 'pt_calibration.GolayControllerManifest',),
    ],
)


#################################################################################
## Noise exposure
#################################################################################
ParadigmDescription(
    'noise_exposure', 'Noise exposure', 'cohort', [
        (PATH + 'noise_exposure.ControllerManifest',),
    ],
)


#################################################################################
## Behavior
#################################################################################
ParadigmDescription(
    'appetitive_gonogo_food', 'Appetitive GO-NOGO food', 'animal', [
        (PATH + 'appetitive.ControllerManifest',),
        (PATH + 'behavior_base.PelletDispenserMixinManifest', True),
    ],
)
