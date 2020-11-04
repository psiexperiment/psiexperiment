import logging
log = logging.getLogger(__name__)

import pickle as pickle
from copy import deepcopy

import numpy as np

from atom.api import Typed, Bool, Str, observe, Property
from enaml.application import deferred_call
from enaml.layout.api import InsertItem, InsertTab
from enaml.workbench.plugin import Plugin

from psi.core.enaml.api import load_manifests
from ..util import get_tagged_values
from .context_item import (
    ContextItem, ContextGroup, Expression, Parameter, ContextMeta
)

from .expression import ExpressionNamespace
from .selector import BaseSelector
from .symbol import Symbol

SELECTORS_POINT = 'psi.context.selectors'
SYMBOLS_POINT = 'psi.context.symbols'
ITEMS_POINT = 'psi.context.items'


def get_preferences(obj):
    return deepcopy(get_tagged_values(obj, 'preference'))


class ContextLookup:

    def __init__(self, context_plugin):
        self.__context_plugin = context_plugin

    def __getattr__(self, name):
        value = self.__context_plugin.get_value(name)
        return value


context_initialized_error = '''
Context not initialized

Your experiment must call the `psi.context.initialize` command at the
appropriate time (usually in response to the `experiment_initialize` action).
See the manual on creating your own experiment if you need further guidance.
'''

class ContextPlugin(Plugin):
    '''
    Plugin that provides a sequence of values that can be used by a controller
    to determine the experiment context_items.
    '''
    context_groups = Typed(dict, {})
    context_items = Typed(dict, {})
    context_meta = Typed(dict, {})
    context_expressions = Typed(list, [])

    selectors = Typed(dict, ())
    symbols = Typed(dict, ())

    # Reflects state of selectors and context_items as currently applied.
    _context_item_state = Typed(dict, ())
    _selector_state = Typed(dict, ())
    _selectors = Typed(dict, ())

    changes_pending = Bool(False)

    # Used to track whether context has properly been initialized. Since all
    # experiments must explicitly initialize context, this is very common to
    # forget. Need to be able to check this to provide better error message.
    initialized = Bool(False)

    _iterators = Typed(dict, ())
    _namespace = Typed(ExpressionNamespace, ())
    _prior_values = Typed(list, ())

    # Subset of context_items that are parameters
    parameters = Property()

    # Return expressions for non-roved parameters
    expressions = Property()

    # Return all expressions, including those for roved parameters
    all_expressions = Property()

    lookup = Typed(ContextLookup, ())

    def _default_lookup(self):
        return ContextLookup(self)

    def start(self):
        self._refresh_selectors()
        self._refresh_items()
        self._refresh_symbols()
        self._bind_observers()

    def stop(self):
        self._unbind_observers()

    def _refresh_selectors(self, event=None):
        # Hidden here to avoid circular import since selectors define a
        # reference to the context plugin.
        log.debug('Refreshing selectors')
        selectors = {}
        point = self.workbench.get_extension_point(SELECTORS_POINT)
        for extension in point.extensions:
            for selector in extension.get_children(BaseSelector):
                selectors[selector.name] = selector
                selector.load_manifest(self.workbench)
        self.selectors = selectors

    def _refresh_symbols(self, event=None):
        symbols = {}
        point = self.workbench.get_extension_point(SYMBOLS_POINT)
        for extension in point.extensions:
            for symbol in extension.get_children(Symbol):
                symbols[symbol.name] = symbol.get_object()
        self.symbols = symbols

    def _refresh_items(self, event=None):
        context_groups = {}
        context_items = {}
        context_meta = {}
        context_expressions = {}
        items = []
        groups = []
        metas = []
        expressions = []

        # First, find all ContextItem, ContextGroup, ContextMeta and
        # Expressions.
        point = self.workbench.get_extension_point(ITEMS_POINT)
        for extension in point.extensions:
            groups.extend(extension.get_children(ContextGroup))
            items.extend(extension.get_children(ContextItem))
            metas.extend(extension.get_children(ContextMeta))
            expressions.extend(extension.get_children(Expression))

        # Now, loop through the groups and find all ContextItems defined under
        # the group. If the group has already been defined in another
        # contribution, raise an error and exit.
        groups_added = []
        for group in groups:
            if group.name in context_groups:
                raise ValueError(f'Context group {group.name} already defined')
            groups_added.append(group.name)
            context_groups[group.name] = group
            group.items = []
            for item in group.children:
                item.set_group(group)
        log.debug('Added context groups: %s', ', '.join(groups_added))

        # Now, go through all "orphan" context items where the group has not
        # been assigned yet.
        for item in items:
            if item.group_name not in context_groups:
                m = f'Missing group "{item.group_name}" for item {item.name}'
                raise ValueError(m)
            item.set_group(context_groups[item.group_name])

        # Now, create a dictionary of all context items. The groups are just
        # for display purposes. Internally, all context items are treated
        # equally.
        for group in context_groups.values():
            for item in group.items:
                if item.name in context_items:
                    m = f'Context item {item.name} already defined'
                    raise ValueError(m)
                context_items[item.name] = item

        for meta in metas:
            context_meta[meta.name] = meta

        for expression in expressions:
            try:
                item = context_items.pop(expression.parameter)
                item.set_group(None)
            except KeyError as e:
                log.warn('%s referenced by expression %s does not exist',
                         expression.parameter, expression.expression)

        load_manifests(context_groups.values(), self.workbench)
        self.context_expressions = expressions
        self.context_items = context_items
        self.context_groups = context_groups
        self.context_meta = context_meta

    def _bind_observers(self):
        self.workbench.get_extension_point(SELECTORS_POINT) \
            .observe('extensions', self._refresh_selectors)
        self.workbench.get_extension_point(ITEMS_POINT) \
            .observe('extensions', self._refresh_items)
        self.workbench.get_extension_point(SYMBOLS_POINT) \
            .observe('extensions', self._refresh_symbols)

    def _unbind_observers(self):
        self.workbench.get_extension_point(SELECTORS_POINT) \
            .unobserve('extensions', self._refresh_selectors)
        self.workbench.get_extension_point(ITEMS_POINT) \
            .unobserve('extensions', self._refresh_items)
        self.workbench.get_extension_point(SYMBOLS_POINT) \
            .unobserve('extensions', self._refresh_symbols)

    @observe('context_items')
    def _bind_context_items(self, change):
        # Note that theoretically we shouldn't have to check if the item is
        # roving or not, but in the event that we change one of the output
        # tokens (which contribute their own set of parameters), the ID of the
        # parameters may change (even if we're just reloading the token).
        # Perhaps there's a more intelligent approach?
        oldvalue = change.get('oldvalue', {})
        newvalue = change.get('value', {})
        for i in oldvalue.values():
            i.unobserve('expression', self._observe_item_updated)
            i.unobserve('rove', self._observe_item_rove)
            if getattr(i, 'rove', False):
                self.unrove_item(i)
        for i in newvalue.values():
            i.observe('expression', self._observe_item_updated)
            i.observe('rove', self._observe_item_rove)
            if getattr(i, 'rove', False):
                self.rove_item(i)

    @observe('symbols')
    def _update_selectors(self, event):
        for selector in self.selectors.values():
            selector.symbols = self.symbols.copy()

    def _observe_item_updated(self, event):
        self._check_for_changes()

    def _observe_item_rove(self, event):
        if event['value']:
            log.debug('Roving {}'.format(event['object'].name))
            self.rove_item(event['object'])
        else:
            log.debug('Unroving {}'.format(event['object'].name))
            self.unrove_item(event['object'])

    @observe('selectors')
    def _bind_selectors(self, change):
        for p in change.get('oldvalue', {}).values():
            p.unobserve('updated', self._observe_selector_updated)
        for p in change.get('value', {}).values():
            p.observe('updated', self._observe_selector_updated)

    def _observe_selector_updated(self, event):
        self._check_for_changes()

    def _get_iterators(self, cycles=np.inf):
        return {k: v.get_iterator(cycles) for k, v in self.selectors.items()}

    def iter_settings(self, iterator='default', cycles=np.inf):
        log.debug('Iterating through settings for %s iterator', iterator)
        # Some paradigms may not actually have an iterator.
        namespace = ExpressionNamespace(self.expressions, self.symbols)
        if iterator:
            selector = self.selectors[iterator].get_iterator(cycles=cycles)
            for setting in selector:
                expressions = {i.name: i.to_expression(e) for i, e in setting.items()}
                namespace.update_expressions(expressions)
                yield namespace.get_values()
                namespace.reset()
        else:
            yield namespace.get_values()

    def unique_values(self, item_name, iterator='default'):
        iterable = self.iter_settings(iterator, 1)
        items = [c[item_name] for c in iterable]
        values = set(items)
        log.debug('Found %d unique values: %r', len(values), values)
        return values

    def get_item(self, item_name):
        return self.context_items[item_name]

    def get_item_info(self, item_name):
        item = self.get_item(item_name)
        return {
            'dtype': item.dtype,
            'label': item.label,
            'compact_label': item.compact_label,
            'default': getattr(item, 'default', None),
            'rove': getattr(item, 'rove', False),
        }

    def rove_item(self, item):
        for selector in self.selectors.values():
            if item not in selector.context_items:
                selector.append_item(item)
        for meta in self.context_meta.values():
            if meta.link_rove:
                meta.add_item(item)

    def unrove_item(self, item):
        for selector in self.selectors.values():
            if item in selector.context_items:
                selector.remove_item(item)
        for meta in self.context_meta.values():
            if meta.link_rove:
                meta.remove_item(item)

    def get_context_info(self):
        return dict((i, self.get_item_info(i)) for i in self.context_items)

    def next(self, save_prior, selector, results):
        '''
        Shortcut for advancing to the next setting.
        '''
        log.debug('Next')
        self.next_setting(save_prior)
        self.next_selector_setting(selector)
        self.set_values(results)

    def next_setting(self, selector=None, save_prior=True):
        '''
        Load next set of expressions. If there are no selectors defined, then
        this essentially clears the namespace and allows expresssions to be
        recomputed.
        '''
        log.debug('Loading next setting')
        if save_prior:
            prior_values = self._prior_values[:]
            prior_values.append(self.get_values())
            self._prior_values = prior_values
        self._namespace.reset()

        if selector is None:
            return
        try:
            log.debug('Configuring next setting from selector %s', selector)
            expressions = next(self._iterators[selector])
            expressions = {i.name: e for i, e in expressions.items()}
            self._namespace.update_expressions(expressions)
        except KeyError:
            m = 'Avaliable selectors include {}'.format(self._iterators.keys())
            log.debug(m)
            raise

    def get_value(self, context_name, trial=None):
        if not self.initialized:
            raise ValueError(context_initialized_error)
        if trial is not None:
            try:
                return self._prior_values[trial][context_name]
            except IndexError:
                return None
        try:
            return self._namespace.get_value(context_name)
        except KeyError as e:
            m = f'{context_name} not defined.'
            raise ValueError(m) from e

    def get_values(self, context_names=None, trial=None):
        if not self.initialized:
            raise ValueError(context_initialized_error)
        if trial is not None:
            return self._prior_values[trial]
        return self._namespace.get_values(names=context_names)

    def set_value(self, context_name, value):
        self._namespace.set_value(context_name, value)

    def set_values(self, values):
        self._namespace.set_values(values)

    def value_changed(self, context_name):
        old = self.get_value(context_name, trial=-1)
        new = self.get_value(context_name)
        return old != new

    def _check_for_changes(self):
        log.debug('Checking for changes')
        for name, state in self._context_item_state.items():
            if name not in self.context_items:
                log.debug('%s not in context item state. changes pending.', name)
                self.changes_pending = True
                return
            item = self.context_items[name]
            if (item.rove, item.expression) != state:
                log.debug('%s expression/rove does not match state. changes pending.', name)
                self.changes_pending = True
                return

        self.changes_pending = self.get_gui_selector_state() != self._selector_state
        if self.changes_pending:
            log.debug('Selectors do not match. Changes pending.')

    def apply_changes(self, cycles=np.inf):
        self._apply_context_item_state()
        self._apply_selector_state()
        self._namespace.update_expressions(self.expressions)
        self._namespace.update_symbols(self.symbols)
        self._iterators = self._get_iterators(cycles)
        self.changes_pending = False
        log.debug('Applied changes')

    def revert_changes(self):
        self._revert_context_item_state()
        self._revert_selector_state()
        self.changes_pending = False

    def _get_parameters(self):
        return {n: i for n, i in self.context_items.items() \
                if isinstance(i, Parameter)}

    def _get_all_expressions(self):
        return {n: i.expression for n, i in self.parameters.items()}

    def _get_expressions(self):
        e = {n: i.expression for n, i in self.parameters.items() if not i.rove}
        e.update({n.parameter: n.expression for n in self.context_expressions})
        return e

    def get_gui_selector_state(self):
        return {n: get_preferences(s) for n, s in self.selectors.items()}

    def _apply_selector_state(self):
        self._selector_state = self.get_gui_selector_state()

    def _revert_selector_state(self):
        for name, state in self._selector_state.items():
            self.selectors[name].__setstate__(deepcopy(state))

    def _apply_context_item_state(self):
        state = {n: get_preferences(i) for n, i in self.context_items.items()}
        self._context_item_state = state

    def _revert_context_item_state(self):
        for name, state in self._context_item_state.items():
            self.context_items[name].__setstate__(deepcopy(state))

    @property
    def has_selectors(self):
        return len(self.selectors) != 0

    def get_parameter(self, name):
        return self.parameters[name]

    def get_meta(self, name):
        return self.context_meta[name]

    def get_metas(self, editable=None):
        values = list(self.context_meta.values())
        if editable is None:
            return values
        return [m for m in values if m.editable == editable]
