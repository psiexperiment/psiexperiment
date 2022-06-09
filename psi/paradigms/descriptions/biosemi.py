from psi.experiment.api import ParadigmDescription


BIOSEMI_PATH = 'psi.paradigms.biosemi.'


ParadigmDescription(
    'biosemi_viz', 'Biosemi Visualization', 'human', [
        {'manifest': BIOSEMI_PATH + 'core.BiosemiVisualizationManifest',
         }
    ]
)


ParadigmDescription(
    'biosemi_nback', 'Biosemi Visualization (N-Back task)', 'human', [
        {'manifest': BIOSEMI_PATH + 'core.BiosemiVisualizationManifest',
         }
    ]
)
