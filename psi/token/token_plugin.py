from atom.api import Unicode
from enaml.workbench.api import Plugin


class TokenPlugin(Plugin):

    base = Unicode()

    def get_context_item(self, item_name):
        context = self.workbench.get_plugin('psi.context')
        return context.get_value('{}_{}'.format(self.base, item_name))
