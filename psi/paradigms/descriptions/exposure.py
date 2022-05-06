from copy import deepcopy
from psi.experiment.api import ParadigmDescription
from .core import CORE_PATH, microphone_mixin, microphone_fft_mixin

PATH = 'psi.paradigms.exposure.'


ParadigmDescription(
    'noise_exposure', 'Noise exposure', 'cohort', [
        {'manifest': PATH + 'noise_exposure.NoiseControllerManifest'},
        microphone_mixin,
        microphone_fft_mixin,
    ],
)
