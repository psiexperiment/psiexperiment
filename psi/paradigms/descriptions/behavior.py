from psi.experiment.api import ParadigmDescription


PATH = 'psi.paradigms.behavior.'
CORE_PATH = 'psi.paradigms.core.'


ParadigmDescription(
    'appetitive-gonogo-food', 'Appetitive GO-NOGO food', 'animal', [
        (PATH + 'behavior_np_gonogo.BehaviorManifest',),
        (PATH + 'behavior_mixins.PelletDispenser',),
    ],
)


ParadigmDescription(
    'auto-gonogo', 'Auto GO-NOGO', 'animal', [
        (PATH + 'behavior_auto_gonogo.BehaviorManifest',),
        (PATH + 'behavior_mixins.BaseGoNogoMixin',),
        (PATH + 'behavior_mixins.WaterBolusDispenser',),
        (CORE_PATH + 'microphone_mixins.MicrophoneSignalViewManifest',),
        (CORE_PATH + 'microphone_mixins.MicrophoneFFTViewManifest',),
    ],
)
