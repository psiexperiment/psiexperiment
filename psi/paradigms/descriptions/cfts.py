from psi.experiment.api import ParadigmDescription
from .core import CORE_PATH, microphone_mixin, microphone_fft_mixin


CAL_PATH = 'psi.paradigms.calibration.'
CFTS_PATH = 'psi.paradigms.cfts.'


eeg_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'eeg_view_mixin',
        'name': 'eeg_view',
        'title': 'EEG display',
        'time_span': 2,
        'time_delay': 0.125,
        'source_name': 'eeg_filtered',
        'y_label': 'EEG (V)'
    }
}


temperature_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest',
    'selected': True,
}


efr_microphone_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
    'attrs': {
        'id': 'microphone_fft_view',
        'title': 'Microphone view (PSD)',
        'fft_time_span': 5,
        'source_name': 'microphone',
        'y_label': 'Microphone (dB)'
    }
}


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
    'dpoae_io', 'DPOAE (input-output)', 'ear', [
        {'manifest': CFTS_PATH + 'dpoae_io.DPOAEIOSimpleManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest', 'selected': True},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
    ],
)


ParadigmDescription(
    'efr_sam', 'SAM EFR', 'ear', [
        {'manifest': CFTS_PATH + 'efr.SAMEFRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.SAMEFRInEarCalibrationMixinManifest', 'selected': True},
        temperature_mixin,
        microphone_mixin,
        efr_microphone_fft_mixin,
        eeg_mixin,
    ]
)


ParadigmDescription(
    'efr_ram', 'RAM EFR', 'ear', [
        {'manifest': CFTS_PATH + 'efr.RAMEFRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.RAMEFRInEarCalibrationMixinManifest', 'selected': True},
        temperature_mixin,
        microphone_mixin,
        efr_microphone_fft_mixin,
        eeg_mixin,
    ]
)


ParadigmDescription(
    'memr_interleaved', 'MEMR (interleaved)', 'ear', [
        {'manifest': CFTS_PATH + 'memr.InterleavedMEMRManifest'},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    'inear_speaker_calibration_chirp', 'In-ear speaker calibration (chirp)', 'ear', [
        {'manifest': CAL_PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest'},
        {'manifest': CAL_PATH + 'calibration_mixins.ChirpMixin'},
        {'manifest': CAL_PATH + 'calibration_mixins.ToneValidateMixin'},
    ]
)
