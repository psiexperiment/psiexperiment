from psi.experiment.api import ParadigmDescription


PATH = 'psi.paradigms.calibration.'
CORE_PATH = 'psi.paradigms.core.'


ParadigmDescription(
    'speaker_calibration_golay', 'Speaker calibration (Golay)', 'calibration', [
        (PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',),
        (PATH + 'calibration_mixins.ChannelSettingMixins',),
        (PATH + 'calibration_mixins.GolayMixin',),
        (PATH + 'calibration_mixins.ToneValidateMixin',),
    ]
)


ParadigmDescription(
    'speaker_calibration_chirp', 'Speaker calibration (chirp)', 'calibration', [
        (PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',),
        (PATH + 'calibration_mixins.ChannelSettingMixins',),
        (PATH + 'calibration_mixins.ChirpMixin',),
        (PATH + 'calibration_mixins.ToneValidateMixin',),
    ]
)


ParadigmDescription(
    'speaker_calibration_tone', 'Speaker calibration (tone)', 'calibration', [
        (PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',),
        (PATH + 'calibration_mixins.ChannelSettingMixins',),
        (PATH + 'calibration_mixins.ToneMixin',),
    ]
)


ParadigmDescription(
    'pistonphone_calibration', 'Pistonphone calibration', 'calibration', [
        (PATH + 'pistonphone_calibration.PistonphoneCalibrationManifest',),
    ]
)


ParadigmDescription(
    'amplifier_calibration', 'Amplifier calibration', 'calibration', [
        (PATH + 'amplifier_calibration.AmplifierCalibrationManifest',),
        (PATH + 'calibration_mixins.ChannelSettingMixins',),
        (CORE_PATH + 'microphone_mixins.MicrophoneFFTViewManifest',
         {'required': True}),
        (CORE_PATH + 'microphone_mixins.MicrophoneSignalViewManifest',
         {'required': True}),
    ]
)
