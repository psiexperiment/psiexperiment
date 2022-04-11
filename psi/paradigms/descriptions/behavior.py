from psi.experiment.api import ParadigmDescription


PATH = 'psi.paradigms.behavior.'
CORE_PATH = 'psi.paradigms.core.'


ParadigmDescription(
    'appetitive_gonogo_food', 'Appetitive GO-NOGO food', 'animal', [
        {'manifest': PATH + 'behavior_np_gonogo.BehaviorManifest'},
        {'manifest': PATH + 'behavior_mixins.PelletDispenser'},
    ],
)


ParadigmDescription(
    'auto_gonogo', 'Auto GO-NOGO', 'animal', [
        {'manifest': PATH + 'behavior_auto_gonogo.BehaviorManifest'},
        {'manifest': PATH + 'behavior_mixins.BaseGoNogoMixin'},
        {'manifest': PATH + 'behavior_mixins.WaterBolusDispenser'},
        {'manifest': CORE_PATH + 'video_mixins.PSIVideo'},
        {'manifest': CORE_PATH + 'microphone_mixins.MicrophoneFFTViewManifest',
         'attrs': {'fft_time_span': 1, 'fft_freq_lb': 5, 'fft_freq_ub': 24000, 'y_label': 'Level (dB)'}
         },
    ],
)
