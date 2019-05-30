from enaml.workbench.api import PluginManifest


class MixinManifest(PluginManifest):

    required = d_(Bool(False))
    selected = d_(Bool(False))
    name = d_(Unicode())
    title = d_(Unicode())
