from atom.api import Bool, Property, Typed, Unicode

from enaml.core.api import d_
from enaml.workbench.api import PluginManifest

from .contribution import PSIContribution


class ExperimentManifest(PluginManifest):

    name = d_(Unicode())
    title = d_(Unicode())
    required = d_(Bool(False))

    def _default_name(self):
        return self.id

    def _default_title(self):
        return self.name.replace('_', ' ').capitalize()


class PSIManifest(PluginManifest):

    contribution = Typed(PSIContribution)
    id = Property(cached=True)

    def _get_id(self):
        class_type = self.__class__.__name__
        return f'{self.contribution.name}.{class_type}'
