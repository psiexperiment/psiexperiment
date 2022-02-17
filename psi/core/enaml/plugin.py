import logging
log = logging.getLogger(__name__)

import textwrap

from enaml.workbench.api import Plugin


class PSIPlugin(Plugin):

    def raise_duplicate_error(self, obj, attr_name, extension,
                              error_type=ValueError):
        mesg = f'''
        Could not load "{getattr(obj, attr_name)}" from extension
        "{extension.id}" since a {obj.__class__.__name__} with the same
        {attr_name} has already been registered.
        '''
        raise error_type(textwrap.fill(textwrap.dedent(mesg)))


    def load_plugins(self, point_id, plugin_type, unique_attr):
        log.debug('Loading plugins for extension point %s', point_id)
        point = self.workbench.get_extension_point(point_id)
        items = {}
        for extension in point.extensions:
            log.debug('Found extension %s', extension.id)
            for item in extension.get_children(plugin_type):
                attr = getattr(item, unique_attr)
                log.debug('Found contribution %s', attr)
                if attr in items:
                    self.raise_duplicate_error(item, unique_attr, extension)
                items[attr] = item
        return items
