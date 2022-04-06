from psi.experiment.api import ParadigmDescription


PATH = 'psi.paradigms.calibration.'


ParadigmDescription(
    'pt_calibration_chirp', 'Probe tube calibration (chirp)', 'calibration', [
        (PATH + 'pt_calibration.ChirpControllerManifest',),
    ],
)


ParadigmDescription(
    'pt_calibration_golay', 'Probe tube calibration (golay)', 'calibration', [
        (PATH + 'pt_calibration.GolayControllerManifest',),
    ],
)
