from psi.experiment.api import ParadigmDescription


PATH = 'psi.paradigms.calibration.'
CORE_PATH = 'psi.paradigms.core.'


ParadigmDescription(
    'speaker_calibration_golay', 'Speaker calibration (Golay)', 'calibration', [
        {'manifest': PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest'},
        {'manifest': PATH + 'calibration_mixins.ChannelSettingMixins'},
        {'manifest': PATH + 'calibration_mixins.GolayMixin'},
        {'manifest': PATH + 'calibration_mixins.ToneValidateMixin'},
    ]
)


ParadigmDescription(
    'speaker_calibration_chirp', 'Speaker calibration (chirp)', 'calibration', [
        {'manifest': PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest'},
        {'manifest': PATH + 'calibration_mixins.ChannelSettingMixins'},
        {'manifest': PATH + 'calibration_mixins.ChirpMixin'},
        {'manifest': PATH + 'calibration_mixins.ToneValidateMixin'},
    ]
)


ParadigmDescription(
    'speaker_calibration_tone', 'Speaker calibration (tone)', 'calibration', [
        {'manifest': PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest'},
        {'manifest': PATH + 'calibration_mixins.ChannelSettingMixins'},
        {'manifest': PATH + 'calibration_mixins.ToneMixin'},
    ]
)


ParadigmDescription(
    'pistonphone_calibration', 'Pistonphone calibration', 'calibration', [
        {'manifest': PATH + 'pistonphone_calibration.PistonphoneCalibrationManifest'},
        {'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
         'required': True,
         'attrs': {'source_name': 'hw_ai', 'time_span': 8, 'y_label': 'PSD (dB re 1V)'},
         },
        {'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
         'required': True,
         'attrs': {'source_name': 'hw_ai', 'y_label': 'PSD (dB re 1V)'}
         },
    ]
)


ParadigmDescription(
    'amplifier_calibration', 'Amplifier calibration', 'calibration', [
        {'manifest': PATH + 'amplifier_calibration.AmplifierCalibrationManifest'},
        {'manifest': PATH + 'calibration_mixins.ChannelSettingMixins'},
        {'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
         'required': True
         },
        {'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
         'required': True
         },
    ]
)
