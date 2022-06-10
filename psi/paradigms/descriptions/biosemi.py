from psi.experiment.api import ParadigmDescription


BIOSEMI_PATH = 'psi.paradigms.biosemi.'
CORE_PATH = 'psi.paradigms.core.'


ParadigmDescription(
    'biosemi_viz', 'Biosemi Visualization', 'human', [
        {'manifest': BIOSEMI_PATH + 'core.BiosemiVisualizationManifest'},
        {'manifest': CORE_PATH + 'websocket_mixins.WebsocketManifest',
         'required': True},
    ]
)


ParadigmDescription(
    'biosemi_nback', 'Biosemi Visualization (N-Back task)', 'human', [
        {'manifest': BIOSEMI_PATH + 'core.BiosemiVisualizationManifest'},
        {'manifest': CORE_PATH + 'websocket_mixins.WebsocketManifest',
         'required': True},
    ]
)
