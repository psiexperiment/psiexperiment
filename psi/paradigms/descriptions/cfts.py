from psi.experiment.api import ParadigmDescription


CAL_PATH = 'psi.paradigms.calibration.'
CORE_PATH = 'psi.paradigms.core.'
CFTS_PATH = 'psi.paradigms.cfts.'


microphone_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'microphone_signal_view',
        'title': 'Microphone view (time)',
        'time_span': 4,
        'time_delay': 0.125,
        'source_name': 'microphone',
        'y_label': 'Microphone (dB)'
    },
}


microphone_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
    'attrs': {
        'id': 'microphone_fft_view',
        'title': 'Microphone view (PSD)',
        'fft_time_span': 0.25,
        'fft_freq_lb': 5,
        'fft_freq_ub': 50000,
        'source_name': 'microphone',
        'y_label': 'Microphone (dB)'
    }
}


eeg_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'eeg_view_mixin',
        'name': 'eeg_view',
        'title': 'EEG display',
        'time_span': 2,
        'time_delay': 0.125,
        'source_name': 'eeg_filtered',
        'y_label': 'EEG (dB)'
    }
}


temperature_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest',
    'selected': True,
}


ParadigmDescription(
    # This is a much more flexible, more powerful ABR experiment interface.
    'abr_io_editable', 'Configurable ABR (input-output)', 'ear', [
        {'manifest': CFTS_PATH + 'abr_io.ABRIOManifest'},
        temperature_mixin,
        eeg_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    # This is the default, simple ABR experiment that most users will want.  
    'abr_io', 'ABR (input-output)', 'ear', [
        {'manifest': CFTS_PATH + 'abr_io.ABRIOSimpleManifest'},
        temperature_mixin,
        eeg_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    'dpoae_io', 'DPOAE input-output', 'ear', [
        {'manifest': CFTS_PATH + 'dpoae_io.DPOAEIOSimpleManifest'},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    'efr', 'EFR (SAM)', 'ear', [
        {'manifest': CFTS_PATH + 'efr.EFRManifest'},
        temperature_mixin,
        eeg_mixin,
        microphone_mixin,
        microphone_fft_mixin,
    ],
)


ParadigmDescription(
    'inear_speaker_calibration_chirp', 'In-ear speaker calibration (chirp)', 'ear', [
        {'manifest': CAL_PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest'},
        {'manifest': CAL_PATH + 'calibration_mixins.ChirpMixin'},
        {'manifest': CAL_PATH + 'calibration_mixins.ToneValidateMixin'},
    ]
)
