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
        'y_label': 'EEG (dB)'
    }
}


temperature_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest',
    'selected': True,
}


#ParadigmDescription(
#    # This is a much more flexible, more powerful ABR experiment interface.
#    'abr_io_editable', 'Configurable ABR (input-output)', 'ear', [
#        {'manifest': CFTS_PATH + 'abr_io.ABRIOManifest'},
#        temperature_mixin,
#        eeg_mixin,
#        {'manifest': CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest', 'selected': True},
#    ]
#)


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
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest', 'selected': True},
    ]
)


base_mixins = [temperature_mixin, microphone_mixin, microphone_fft_mixin]
efr_mixins = base_mixins + [eeg_mixin]


ParadigmDescription(
    'efr_sam', 'SAM EFR', 'ear', [
        {'manifest': CFTS_PATH + 'efr.SAMEFRManifest'},
    ] + efr_mixins,
)


ParadigmDescription(
    'efr_ram', 'RAM EFR', 'ear', [
        {'manifest': CFTS_PATH + 'efr.RAMEFRManifest'},
    ] + efr_mixins,
)



ParadigmDescription(
    'memr_simultaneous', 'MEMR (simultaneous)', 'ear', [
        {'manifest': CFTS_PATH + 'memr.SimultaneousMEMRManifest'},
    ] + base_mixins,
)


ParadigmDescription(
    'memr_interleaved', 'MEMR (interleaved)', 'ear', [
        {'manifest': CFTS_PATH + 'memr.InterleavedMEMRManifest'},
    ] + base_mixins,
)


ParadigmDescription(
    'inear_speaker_calibration_chirp', 'In-ear speaker calibration (chirp)', 'ear', [
        {'manifest': CAL_PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest'},
        {'manifest': CAL_PATH + 'calibration_mixins.ChannelSettingMixins'},
        {'manifest': CAL_PATH + 'calibration_mixins.ChirpMixin'},
        {'manifest': CAL_PATH + 'calibration_mixins.ToneValidateMixin'},
    ]
)
