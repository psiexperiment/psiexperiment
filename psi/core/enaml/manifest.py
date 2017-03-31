from atom.api import Typed, Unicode, Property

from enaml.workbench.api import PluginManifest

from .contribution import PSIContribution


class PSIManifest(PluginManifest):

    contribution = Typed(PSIContribution)
    base_id = Unicode()
    id = Property(cached=True)

    def _get_id(self):
        if self.base_id:
            return unicode('.'.join((self.base_id, self.contribution.name)))
        else:
            return unicode(self.contribution.name)
