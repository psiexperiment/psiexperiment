CORE_PATH = 'psi.paradigms.core.'


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

