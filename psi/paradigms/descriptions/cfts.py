from psi.experiment.api import ParadigmDescription


CAL_PATH = 'psi.paradigms.calibration.'
CORE_PATH = 'psi.paradigms.core.'
CFTS_PATH = 'psi.paradigms.cfts.'


ParadigmDescription(
    # This is a much more flexible, more powerful ABR experiment interface.
    'abr_io_editable', 'Configurable ABR (input-output)', 'ear', [
        {'manifest': CFTS_PATH + 'abr_io.ABRIOManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest', 'selected': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.EEGViewMixinManifest', 'selected': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    # This is the default, simple ABR experiment that most users will want.  
    'abr_io', 'ABR (input-output)', 'ear', [
        {'manifest': CFTS_PATH + 'abr_io.ABRIOSimpleManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest', 'selected': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.EEGViewMixinManifest', 'selected': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    'dpoae_io', 'DPOAE input-output', 'ear', [
        {'manifest': CFTS_PATH + 'dpoae_io.DPOAEIOSimpleManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest', 'selected': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest', 'selected': True},
        {'manifest': CORE_PATH + 'microphone_mixins.MicrophoneSignalViewManifest', 'selected': True},
        {'manifest': CORE_PATH + 'microphone_mixins.MicrophoneFFTViewManifest',
         'selected': True,
         'attrs': {'source_name': 'microphone', 'fft_time_span': 0.25}
         },
    ]
)


ParadigmDescription(
    'efr', 'EFR (SAM)', 'ear', [
        {'manifest': CFTS_PATH + 'efr.EFRManifest'},
        {'manifest': CORE_PATH + 'microphone_mixins.MicrophoneSignalViewManifest', 'selected': True},
        {'manifest': CORE_PATH + 'microphone_mixins.MicrophoneFFTViewManifest', 'selected': True},
    ],
)


ParadigmDescription(
    'inear_speaker_calibration_chirp', 'In-ear speaker calibration (chirp)', 'ear', [
        {'manifest': CAL_PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest'},
        {'manifest': CAL_PATH + 'calibration_mixins.ChirpMixin'},
        {'manifest': CAL_PATH + 'calibration_mixins.ToneValidateMixin'},
    ]
)
