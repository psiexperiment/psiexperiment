import logging
log = logging.getLogger(__name__)

import textwrap

from enaml.workbench.api import Plugin

from .util import load_manifests


class PSIPlugin(Plugin):

    def raise_duplicate_error(self, obj, attr_name, extension, orig_extension,
                              error_type=ValueError):
        mesg = f'''
        Could not load "{getattr(obj, attr_name)}" from extension
        "{extension.id}" since a {obj.__class__.__name__} with the same
        {attr_name} has already been registered by {orig_extension.id}.".
        '''
        raise error_type(textwrap.fill(textwrap.dedent(mesg)))


    def load_plugins(self, point_id, plugin_type, unique_attr, **factory_kw):
        '''
        Load plugins for extension point

        Parameters
        ----------
        point_id : str
            ID of the extension point
        plugin_type : class
            The class of the plugin to search for. Subclasses will also be
            loaded.
        unique_attr : str
            The name of the instance attribute that must be unique across all
            instances.

        Remaining keyword arguments are passed to any factories found on the
        contributions to the extension point.
        '''
        log.debug('Loading plugins for extension point %s', point_id)
        point = self.workbench.get_extension_point(point_id)
        items = {}
        # Track the original source of the attribute that way we can provide
        # more informative error message if we have a duplicate.
        item_source = {}
        for extension in point.extensions:
            log.debug('... Found extension %s', extension.id)
            children = extension.get_children(plugin_type)
            if extension.factory is not None:
                children.extend(extension.factory(**factory_kw))
            for item in children:
                attr = getattr(item, unique_attr)
                log.debug('... ... found contribution %s', attr)
                if attr in items:
                    self.raise_duplicate_error(item, unique_attr, extension, item_source[attr])
                items[attr] = item
                item_source[attr] = extension

        return items

    def load_manifests(self, items):
        load_manifests(items, self.workbench)
