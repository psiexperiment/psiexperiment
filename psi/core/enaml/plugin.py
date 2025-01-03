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

        Returns
        -------
        items : dict of dict
            Returns a dictionary mapping Plugin class to a dictionary mapping
            item name to the item instance for each instance of that Plugin
            class.
        '''
        plugin_info = {plugin_type: unique_attr}
        result = self.load_multiple_plugins(point_id, plugin_info, **factory_kw)
        return result.get(plugin_type, {})

    def load_multiple_plugins(self, point_id, plugin_info, **factory_kw):
        '''
        Load multiple plugins for extension point

        Parameters
        ----------
        point_id : str
            ID of the extension point
        plugin_info : dict
            Dictionary mapping Plugin class to the unique attribute on the
            class (e.g., context item name) that is used to track the instance
            of the plugin in psiexperiment.

        Remaining keyword arguments are passed to any factories found on the
        contributions to the extension point.

        Returns
        -------
        items : dict of dict
            Returns a dictionary mapping Plugin class to a dictionary mapping
            item name to the item instance for each instance of that Plugin
            class.
        '''
        log.debug('Loading plugins for extension point %s', point_id)
        point = self.workbench.get_extension_point(point_id)

        items = {}

        # Track the original source of the attribute that way we can provide
        # more informative error message if we have a duplicate.
        item_source = {}

        # Generate a list of all children across the extensions
        children = []
        for extension in point.extensions:
            log.debug('... Found extension %s', extension.id)
            children.extend(extension.children)
            if extension.factory is not None:
                children.extend(extension.factory(**factory_kw))

        # Now, group together the items into their respective plugins.
        for plugin_type, unique_attr in plugin_info.items():
            for item in children:
                plugin_items = items.setdefault(plugin_type, {})
                plugin_item_source = item_source.setdefault(plugin_type, {})
                if isinstance(item, plugin_type):
                    attr = getattr(item, unique_attr)
                    log.debug('... ... found contribution %s', attr)

                    if attr in plugin_items:
                        self.raise_duplicate_error(item,
                                                   unique_attr,
                                                   extension,
                                                   plugin_item_source[attr])

                    plugin_items[attr] = item
                    plugin_item_source[attr] = extension

        return items

    def load_manifests(self, items):
        load_manifests(items, self.workbench)
