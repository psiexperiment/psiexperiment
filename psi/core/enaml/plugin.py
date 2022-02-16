import textwrap

from enaml.workbench.api import Plugin


class PSIPlugin(Plugin):

    def raise_duplicate_error(self, obj, attr_name, extension,
                              plugin_type, error_type=ValueError):
        mesg = f'''
        Could not load "{getattr(obj, attr_name)}" from extension
        "{extension.id}" since a {ob.__class__.__name__} with the same
        {attr_name} has already been registered.
        '''
        raise error_type(textwrap.fill(textwrap.dedent(mesg)))
