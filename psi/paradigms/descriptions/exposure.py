from psi.experiment.api import ParadigmDescription
from .core import CORE_PATH


PATH = 'psi.paradigms.exposure.'


microphone_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'microphone_signal_view',
        'title': 'Microphone view (time)',
        'time_span': 10,
        'time_delay': 0,
        'source_name': 'microphone_filtered',
        'y_label': 'Microphone (dB)'
    },
}


microphone_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
    'attrs': {
        'id': 'microphone_fft_view',
        'title': 'Microphone view (PSD)',
        'fft_time_span': 1,
        'waveform_averages': 10,
        'fft_freq_lb': 500,
        'fft_freq_ub': 64000,
        'source_name': 'microphone_filtered',
        'y_label': 'Microphone (dB)'
    }
}


ParadigmDescription(
    'noise_exposure', 'Noise exposure', 'cohort', [
        {'manifest': PATH + 'noise_exposure.NoiseControllerManifest'},
        microphone_mixin,
        microphone_fft_mixin,
    ],
)
