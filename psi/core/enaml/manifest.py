from atom.api import Typed, Property

from enaml.workbench.api import PluginManifest

from .contribution import PSIContribution


class PSIManifest(PluginManifest):

    contribution = Typed(PSIContribution)
    id = Property(cached=True)

    def _get_id(self):
        class_type = self.__class__.__name__
        return f'{self.contribution.name}.{class_type}'
