from atom.api import List
from enaml.core.api import d_
from enaml.workbench.api import PluginManifest


class PSIManifest(PluginManifest):

    supplements = d_(List())
    requires = d_(List())
