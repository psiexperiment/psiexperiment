from psi.experiment.api import ParadigmDescription


PATH = 'psi.paradigms.calibration.'


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
