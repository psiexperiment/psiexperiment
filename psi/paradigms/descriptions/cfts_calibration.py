from psi.experiment.api import ParadigmDescription


PATH = 'psi.paradigms.calibration.'
CORE_PATH = 'psi.paradigms.core.'


#ParadigmDescription(
#    'pt_calibration_chirp', 'Probe tube calibration (chirp)', 'calibration', [
#        {'manifest': PATH + 'pt_calibration.ChirpControllerManifest',},
#    ],
#)


ParadigmDescription(
    'pt_calibration_golay', 'Probe tube calibration (golay)', 'calibration', [
        {'manifest': PATH + 'pt_calibration.BasePTCalibrationManifest',},
        {'manifest': PATH + 'pt_calibration.PTGolayMixin',},
        {'manifest': PATH + 'calibration_mixins.ToneValidateMixin',},
    ],
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
        {'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
         'required': True
         },
        {'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
         'required': True
         },
    ]
)
