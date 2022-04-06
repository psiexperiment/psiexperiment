from psi.experiment.api import ParadigmDescription

PATH = 'psi.paradigms.exposure.'


ParadigmDescription(
    'noise_exposure', 'Noise exposure', 'cohort', [
        (PATH + 'noise_exposure.NoiseControllerManifest',),
    ],
)
