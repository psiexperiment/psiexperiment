import cPickle as pickle
from copy import deepcopy

from atom.api import Typed, Bool, Str, observe
from enaml.application import deferred_call
from enaml.layout.api import InsertItem, InsertTab
from enaml.workbench.plugin import Plugin

from .context_item import ContextItem, Parameter
from .context_group import ContextGroup
from .expression import ExpressionNamespace

SELECTORS_POINT = 'psi.context.selectors'
ITEMS_POINT = 'psi.context.items'
#SYMBOLS_POINT = 'psi.context.symbols'


def copy_attrs(from_atom, to_atom):
    if to_atom.__class__ != from_atom.__class__:
        raise ValueError()
    for name, member in to_atom.members().items():
        if member.metadata and member.metadata.get('transient', False):
            continue
        try:
            value = deepcopy(getattr(from_atom, name))
            setattr(to_atom, name, value)
        except:
            pass


class ContextPlugin(Plugin):
    '''
    Plugin that provides a sequence of values that can be used by a controller
    to determine the experiment context_items.
    '''
    context_groups = Typed(dict, {})
    context_items = Typed(dict, {})
    roving_items = Typed(list, [])

    selectors = Typed(dict, ())
    symbols = Typed(dict, ())

    # Reflects state of selectors and context_items as currently applied.
    _selectors = Typed(dict, ())
    _context_items = Typed(dict, ())
    _roving_items = Typed(list, ())

    changes_pending = Bool(False)

    _iterators = Typed(dict, ())
    _namespace = Typed(ExpressionNamespace, ())
    _prior_values = Typed(list, ())

    def start(self):
        self._refresh_selectors()
        self._refresh_items()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _refresh_selectors(self):
        # Hidden here to avoid circular import since selectors define a
        # reference to the context plugin.
        from .selector import BaseSelector
        selectors = {}
        point = self.workbench.get_extension_point(SELECTORS_POINT)
        for extension in point.extensions:
            for selector in extension.get_children(BaseSelector):
                selectors[selector.name] = selector
        self.selectors = selectors

    def _refresh_items(self, event=None):
        context_groups = {}
        context_items = {}

        point = self.workbench.get_extension_point(ITEMS_POINT)
        for extension in point.extensions:
            items = extension.get_children(ContextItem)
            groups = extension.get_children(ContextGroup)
            factory = getattr(extension, 'factory')
            if factory is not None:
                items.extend(factory('items'))
                groups.extend(factory('groups'))

            for group in groups:
                if group.name in context_groups:
                    m = 'Context group {} already defined'.format(group.name)
                    raise ValueError(m)
                context_groups[group.name] = group

            for item in items:
                if item.name in context_items:
                    m = 'Context item {} already defined'.format(item.name)
                    raise ValueError(m)
                context_items[item.name] = item

        # Now that everything has been loaded, check to make sure we have no
        # missing groups.
        for item in context_items.values():
            if item.group not in context_groups:
                m = 'Group {} for {} does not exist'
                m = m.format(item.group, item.name)
                raise ValueError(m)

        self.context_items = context_items
        self.context_groups = context_groups

    def _bind_observers(self):
        self.workbench.get_extension_point(SELECTORS_POINT) \
            .observe('extensions', self._refresh_selectors)
        self.workbench.get_extension_point(ITEMS_POINT) \
            .observe('extensions', self._refresh_items)

    def _unbind_observers(self):
        self.workbench.get_extension_point(SELECTORS_POINT) \
            .unobserve('extensions', self._refresh_all)

    @observe('context_items')
    def _bind_context_items(self, change):
        for i in change.get('oldvalue', {}).values():
            if hasattr(i, 'expression'):
                i.unobserve('expression', self._observe_item_expression)
        for i in change.get('value', {}).values():
            if hasattr(i, 'expression'):
                i.observe('expression', self._observe_item_expression)

    def _observe_item_expression(self, event):
        self._check_for_changes()

    @observe('selectors')
    def _bind_selectors(self, change):
        for p in change.get('oldvalue', {}).values():
            p.unobserve('updated', self._observe_selector_updated)
        for p in change.get('value', {}).values():
            p.observe('updated', self._observe_selector_updated)
            p.context_plugin = self

    def _observe_selector_updated(self, event):
        self._check_for_changes()

    def _update_attrs(self, context_items, selectors, roving_items):
        for i in self.context_items:
            if i not in context_items:
                # TODO: how to handle this? warning?
                continue
            from_items = context_items[i]
            to_items = self.context_items[i]
            copy_attrs(from_items, to_items)
        for s in self.selectors:
            from_selector = selectors[s]
            to_selector = self.selectors[s]
            copy_attrs(from_selector, to_selector)
        self.roving_items = roving_items

    def _check_for_changes(self):
        ci_changed = self.context_items != self._context_items
        s_changed = self.selectors != self._selectors
        ri_changed = self.roving_items != self._roving_items
        self.changes_pending = ci_changed or s_changed or ri_changed

    def _get_expressions(self):
        # Return a dictionary of expressions for all context_items that are not
        # managed by the selectors.
        expressions = {}
        for item_name, item in self._context_items.items():
            if isinstance(item, Parameter) and \
                    item_name not in self._roving_items:
                expressions[item_name] = item.expression
        return expressions

    def _get_sequences(self):
        return dict((n, s.__getstate__()) for n, s in self.selectors.items())

    def _get_iterators(self):
        return dict((k, v.get_iterator()) for k, v in self.selectors.items())

    def get_item_info(self, item_name):
        item = self.context_items[item_name]
        return {
            'dtype': item.dtype,
            'label': item.label,
            'compact_label': item.compact_label,
            'default': getattr(item, 'default', None),
            'rove': item_name in self._roving_items,
        }

    def rove_item(self, item_name):
        if self.context_items[item_name].scope != 'trial':
            raise ValueError('Cannot rove {}'.format(item_name))
        roving_items = self.roving_items[:]
        roving_items.append(item_name)
        self.roving_items = roving_items
        for selector in self.selectors.values():
            if item_name not in selector.context_items:
                selector.append_item(item_name)

    def unrove_item(self, item_name):
        roving_items = self.roving_items[:]
        roving_items.remove(item_name)
        self.roving_items = roving_items
        for selector in self.selectors.values():
            if item_name in selector.context_items:
                selector.remove_item(item_name)

    def get_context_info(self):
        return dict((i, self.get_item_info(i)) for i in self.context_items)

    def next(self, save_prior, selector, results):
        '''
        Shortcut for advancing to the next setting.
        '''
        self.next_setting(save_prior)
        self.next_selector_setting(selector)
        self.set_values(results)

    def next_setting(self, selector, save_prior):
        '''
        Load next set of expressions from the specified sequence
        '''
        if save_prior:
            self._prior_values.append(self.get_values())
        self._namespace.reset()
        expressions = self._iterators[selector].next()
        self._namespace.update_expressions(expressions)

    def get_value(self, context_name, trial=None, fail_mode='error'):
        if trial is not None:
            try:
                return self._prior_values[trial][context_name]
            except IndexError:
                return None
        try:
            return self._namespace.get_value(context_name)
        except KeyError:
            if fail_mode == 'error':
                raise
            elif fail_mode == 'ignore':
                return None
            elif fail_mode == 'default':
                return self.context_items[context_name].default
            else:
                raise ValueError('Unsupported fail mode {}'.format(fail_mode))

    def get_values(self, trial=None):
        if trial is not None:
            return self._prior_values[trial]
        return self._namespace.get_values()

    def set_value(self, context_name, value):
        self._namespace.set_value(context_name, value)

    def set_values(self, values):
        self._namespace.set_values(values)

    def value_changed(self, context_name):
        old = self.get_value(context_name, trial=-1)
        new = self.get_value(context_name)
        return old != new

    def apply_changes(self):
        self._context_items = deepcopy(self.context_items)
        self._selectors = deepcopy(self.selectors)
        self._roving_items = deepcopy(self.roving_items)

        self._namespace.update_expressions(self._get_expressions())
        self._namespace.update_symbols(self.symbols)
        self._iterators = self._get_iterators()
        self.changes_pending = False

    def revert_changes(self):
        self._update_attrs(self._context_items,
                           self._selectors,
                           self._roving_items)
        self._check_for_changes()
