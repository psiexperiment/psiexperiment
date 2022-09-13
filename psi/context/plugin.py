import logging
log = logging.getLogger(__name__)

from copy import deepcopy
from functools import partial
import itertools
import pickle as pickle

import numpy as np

from atom.api import Typed, Bool, Str, observe, Property
from enaml.application import deferred_call
from enaml.layout.api import InsertItem, InsertTab
from enaml.workbench.plugin import Plugin

from psi.core.enaml.api import load_manifests
from ..util import get_tagged_values
from .context_item import (
    ContextItem, ContextGroup, ContextSet, Expression, Parameter, ContextMeta
)


from psi.core.enaml.api import PSIPlugin
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
        return self.__context_plugin.get_value(name)

    def unique_values(self, item_name, iterator='default'):
        return self.__context_plugin.unique_values(item_name, iterator)

    def lookup(self, attr):
        cb = partial(getattr, self, attr)
        cb.is_lookup = True
        return cb

    def get_selector(self, name='default'):
        return self.__context_plugin.get_selector(name)

    def get_names(self, name='default'):
        return self.__context_plugin.get_names(name)

    def get_range(self, item_name, iterator='default'):
        return self.__context_plugin.get_range(item_name, iterator)


context_initialized_error = '''
Context not initialized

Your experiment must call the `psi.context.initialize` command at the
appropriate time (usually in response to the `experiment_initialize` action).
See the manual on creating your own experiment if you need further guidance.
'''


class ContextPlugin(PSIPlugin):
    '''
    Plugin that provides a sequence of values that can be used by a controller
    to determine the experiment context_items.
    '''
    context_groups = Typed(dict, {})
    context_items = Typed(dict, {})
    context_meta = Typed(dict, {})
    context_expressions = Typed(dict, {})

    # True if some of the context_meta items are user-configurable.
    context_meta_editable = Bool(False)

    selectors = Typed(dict, ())
    symbols = Typed(dict, ())

    # Reflects state of selectors and context_items as currently applied.
    _parameter_state = Typed(dict, ())
    _selector_state = Typed(dict, ())
    _selectors = Typed(dict, ())

    changes_pending = Bool(False)

    # Used to track whether context has properly been initialized. Since all
    # experiments must explicitly initialize context, this is very common to
    # forget. Need to be able to check this to provide better error message.
    initialized = Bool(False)

    _iterators = Typed(dict, ())
    _namespace = Typed(ExpressionNamespace, ())

    # Subset of context_items that are parameters
    parameters = Property()

    # Return expressions for non-roved parameters
    expressions = Property()

    # Return all expressions, including those for roved parameters
    all_expressions = Property()

    lookup = Typed(ContextLookup)

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
                if selector.name in selectors:
                    m = f'Already have a selector named "{selector.name}"'
                    raise ValueError(m)
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
        # Find all plugin contributions
        context_groups = self.load_plugins(ITEMS_POINT, ContextGroup, 'name')
        context_sets = self.load_plugins(ITEMS_POINT, ContextSet, 'name')
        context_items = self.load_plugins(ITEMS_POINT, ContextItem, 'name')
        context_meta = self.load_plugins(ITEMS_POINT, ContextMeta, 'name')
        context_expressions = self.load_plugins(ITEMS_POINT, Expression, 'parameter')

        groups_updated = set()

        # At this point, `context_items` is only the "orphan" context items
        # where the group has not yet been assigned.
        for item in itertools.chain(context_items.values(), context_sets.values()):
            if item.group_name not in context_groups:
                valid_names = ', '.join(context_groups.keys())
                m = f'Missing group "{item.group_name}" for item {item.name}. Valid groups are {valid_names}.'
                raise ValueError(m)
            group = context_groups[item.group_name]
            item.set_parent(group)
            groups_updated.add(group)

        # Now, loop through the groups and find all ContextItems defined under
        # the group. If the group has already been defined in another
        # contribution, raise an error. Also, build up the ContextItems
        # dictionary so that we have a list of all the context items we want to
        # display.
        context_items = {}
        for group in context_groups.values():
            for item in group.children:
                if isinstance(item, ContextItem):
                    if item.name in context_items:
                        m = f'Context item {item.name} already defined'
                        raise ValueError(m)
                    else:
                        context_items[item.name] = item
                elif isinstance(item, ContextSet):
                    if item.name in context_sets.values():
                        m = f'Context set {item.name} already defined'
                        raise ValueError(m)
                    else:
                        context_sets[item.name] = item

        for cset in context_sets.values():
            for item in cset.children:
                if isinstance(item, ContextItem):
                    if item.name in context_items:
                        m = f'Context item {item.name} already defined'
                        raise ValueError(m)
                    else:
                        context_items[item.name] = item

        for expression in context_expressions.values():
            try:
                item = context_items.pop(expression.parameter)
                groups_updated.add(item.parent)
                item.set_parent(None)
            except KeyError as e:
                # It's best to make this an error. Previously I just logged a
                # warning, but this makes debugging more difficult sometimes if
                # you write an expression referencing a parameter that does not
                # exist (e.g., because of a typo or a change to a token name).
                # If, in the future, we need the ability to ignore missing
                # parameters, I would add an attribute to the Expression class
                # indicating that it's OK to ignore if the parameter does not exist.
                raise ValueError(f'{expression.parameter} referenced by'
                                 f'expression {expression.expression} '
                                 'does not exist.')

        load_manifests(context_groups.values(), self.workbench)
        self.context_expressions = context_expressions
        self.context_items = context_items
        self.context_groups = context_groups
        self.context_meta = context_meta
        self.context_meta_editable = len(self.get_metas(editable=True)) > 0

        for group in context_groups.values():
            group.updated = True

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

        # This triggers an interesting "bug" where ContextMeta is set to track
        # roving items and we change one of the tokens. I don't have a great
        # work-around right now.
        oldvalue = change.get('oldvalue', {})
        newvalue = change.get('value', {})
        for i in oldvalue.values():
            i.unobserve('expression', self._item_updated)
            i.unobserve('rove', self._item_roved)
            if getattr(i, 'rove', False):
                if id(i) != id(newvalue.get(i.name, None)):
                    self.unrove_item(i)
        for i in newvalue.values():
            i.observe('expression', self._item_updated)
            i.observe('rove', self._item_roved)
            if getattr(i, 'rove', False):
                if id(i) != id(oldvalue.get(i.name, None)):
                    self.rove_item(i)

    @observe('symbols')
    def _update_selectors(self, event):
        for selector in self.selectors.values():
            selector.symbols = self.symbols.copy()

    def _item_updated(self, event):
        self._check_for_changes()

    def _item_roved(self, event):
        if event['value']:
            log.debug('Roving {}'.format(event['object'].name))
            self.rove_item(event['object'])
        else:
            log.debug('Unroving {}'.format(event['object'].name))
            self.unrove_item(event['object'])

    @observe('selectors')
    def _bind_selectors(self, change):
        log.debug('Binding selectors')
        for p in change.get('oldvalue', {}).values():
            p.unobserve('updated', self._selector_updated)
        for p in change.get('value', {}).values():
            p.observe('updated', self._selector_updated)

    def _selector_updated(self, event):
        log.debug('Selectors updated')
        self._check_for_changes()

    def _get_iterators(self, cycles=np.inf):
        log.debug('Getting iterators')
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

    def n_values(self, iterator='default'):
        iterable = self.iter_settings(iterator, 1)
        return len([c for c in iterable])

    def unique_values(self, item_names, iterator='default'):
        if isinstance(item_names, str):
            item_names = [item_names]
            extract = True
        else:
            extract = False

        values = set()
        for setting in self.iter_settings(iterator, 1):
            if isinstance(item_names, str):
                values.add(setting[item_names])
            else:
                values.add(tuple(setting[n] for n in item_names))
        log.debug('Found %d unique values: %r', len(values), values)

        if extract:
            values = {v[0] for v in values}

        return values

    def get_range(self, item_name, iterator='default'):
        values = self.unique_values(item_name, iterator)
        return min(values), max(values)

    def get_names(self, iterator='default'):
        return [i.name for i in self.get_selector(iterator).context_items]

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
        log.debug('Roving item %r', item)
        for selector in self.selectors.values():
            if item not in selector.context_items:
                selector.append_item(item)
        for meta in self.context_meta.values():
            if meta.link_rove:
                meta.add_item(item)

    def unrove_item(self, item):
        log.debug('Unroving item %r', item)
        for selector in self.selectors.values():
            if item in selector.context_items:
                selector.remove_item(item)
        for meta in self.context_meta.values():
            if meta.link_rove:
                meta.remove_item(item)

    def get_context_info(self):
        return dict((i, self.get_item_info(i)) for i in self.context_items)

    def next_setting(self, selector=None):
        '''
        Load next set of expressions. If there are no selectors defined, then
        this clears the namespace and allows expresssions to be recomputed.
        '''
        log.debug('Loading next setting')
        self._namespace.reset()

        if selector is None:
            return
        try:
            log.info('Configuring next setting from selector %s', selector)
            expressions = next(self._iterators[selector])
            expressions = {i.name: e for i, e in expressions.items()}
            self._namespace.update_expressions(expressions)
        except KeyError:
            m = 'Avaliable selectors include {}'.format(self._iterators.keys())
            log.debug(m)
            raise

    def get_value(self, context_name):
        if not self.initialized:
            raise ValueError(context_initialized_error)
        try:
            return self._namespace.get_value(context_name)
        except KeyError as e:
            m = f'{context_name} not defined.'
            raise ValueError(m) from e

    def get_values(self, context_names=None):
        if not self.initialized:
            raise ValueError(context_initialized_error)
        return self._namespace.get_values(names=context_names)

    def set_value(self, context_name, value):
        self._namespace.set_value(context_name, value)

    def set_values(self, values):
        self._namespace.set_values(values)

    def _check_for_changes(self):
        log.debug('Checking for changes')
        for name, state in self._parameter_state.items():
            if name not in self.parameters:
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
        log.debug('Applying changes')
        self._apply_parameter_state()
        self._apply_selector_state()
        self._namespace.update_expressions(self.expressions)
        self._namespace.update_symbols(self.symbols)
        self._iterators = self._get_iterators(cycles)
        self.changes_pending = False
        log.debug('Applied changes')

    def revert_changes(self):
        log.debug('Reverting changes')
        self._revert_parameter_state()
        self._revert_selector_state()
        self.changes_pending = False

    def _get_parameters(self):
        return {n: i for n, i in self.context_items.items() \
                if isinstance(i, Parameter)}

    def _get_all_expressions(self):
        return {n: i.expression for n, i in self.parameters.items()}

    def _get_expressions(self):
        e = {n: i.expression for n, i in self.parameters.items() if not i.rove}
        e.update({n: i.expression for n, i in self.context_expressions.items()})
        return e

    def get_gui_selector_state(self):
        return {n: get_preferences(s) for n, s in self.selectors.items()}

    def _apply_selector_state(self):
        self._selector_state = self.get_gui_selector_state()

    def _revert_selector_state(self):
        for name, state in self._selector_state.items():
            self.selectors[name].__setstate__(deepcopy(state))

    def _apply_parameter_state(self):
        state = {n: get_preferences(i) for n, i in self.parameters.items()}
        self._parameter_state = state

    def _revert_parameter_state(self):
        for name, state in self._parameter_state.items():
            self.context_items[name].__setstate__(deepcopy(state))

    @property
    def has_selectors(self):
        return len(self.selectors) != 0

    def get_selector(self, name='default'):
        return self.selectors[name]

    def get_parameter(self, name):
        return self.parameters[name]

    def get_meta(self, name):
        log.debug('Getting meta for %s', name)
        return self.context_meta[name]

    def get_metas(self, editable=None):
        log.debug('Getting meta information')
        values = list(self.context_meta.values())
        if editable is None:
            return values
        return [m for m in values if m.editable == editable]
