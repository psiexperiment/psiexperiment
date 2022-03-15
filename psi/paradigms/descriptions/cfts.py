from psi.experiment.api import ParadigmDescription


CAL_PATH = 'psi.paradigms.calibration.'
CORE_PATH = 'psi.paradigms.core.'
CFTS_PATH = 'psi.paradigms.cfts.'


ParadigmDescription(
    # This is a much more flexible, more powerful ABR experiment interface.
    'abr_io_editable', 'Configurable ABR (input-output)', 'ear', [
        (CFTS_PATH + 'abr_io.ABRIOManifest',),
        (CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest', 
         {'selected': True}),
        (CFTS_PATH + 'cfts_mixins.EEGViewMixinManifest', 
         {'selected': True}),
        (CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest',
         {'selected': True}),
    ]
)


ParadigmDescription(
    # This is the default, simple ABR experiment that most users will want.  
    'abr_io', 'ABR (input-output)', 'ear', [
        (CFTS_PATH + 'abr_io.ABRIOSimpleManifest',),
        (CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest',
         {'selected': True}),
        (CFTS_PATH + 'cfts_mixins.EEGViewMixinManifest',
         {'selected': True}),
        (CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest',
         {'selected': True}),
    ]
)


ParadigmDescription(
    'dpoae_io', 'DPOAE input-output', 'ear', [
        (CFTS_PATH + 'dpoae_io.DPOAEIOSimpleManifest',),
        (CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest',
         {'selected': True}),
        (CFTS_PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest',
         {'selected': True}),
        (CORE_PATH + 'microphone_mixins.MicrophoneSignalViewManifest',
         {'selected': True}),
        (CORE_PATH + 'microphone_mixins.MicrophoneFFTViewManifest',
         {'selected': True}),
    ]
)


ParadigmDescription(
    'efr', 'EFR (SAM)', 'ear', [
        (CFTS_PATH + 'efr.EFRManifest',),
        (CORE_PATH + 'microphone_mixins.MicrophoneSignalViewManifest',
         {'selected': True}),
        (CORE_PATH + 'microphone_mixins.MicrophoneFFTViewManifest',
         {'selected': True}),
    ],
)


ParadigmDescription(
    'inear_speaker_calibration_chirp', 'In-ear speaker calibration (chirp)', 'ear', [
        (CAL_PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',),
        (CAL_PATH + 'calibration_mixins.ChirpMixin',),
        (CAL_PATH + 'calibration_mixins.ToneValidateMixin',),
    ]
)
