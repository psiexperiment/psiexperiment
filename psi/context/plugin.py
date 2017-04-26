import logging
log = logging.getLogger(__name__)

import pickle as pickle
from copy import deepcopy

from atom.api import Typed, Bool, Str, observe
from enaml.application import deferred_call
from enaml.layout.api import InsertItem, InsertTab
from enaml.workbench.plugin import Plugin

from .context_item import ContextItem, Parameter, ContextMeta
from .context_group import ContextGroup
from .expression import ExpressionNamespace
from .selector import BaseSelector
from .symbol import Symbol

SELECTORS_POINT = 'psi.context.selectors'
SYMBOLS_POINT = 'psi.context.symbols'
ITEMS_POINT = 'psi.context.items'


class ContextPlugin(Plugin):
    '''
    Plugin that provides a sequence of values that can be used by a controller
    to determine the experiment context_items.
    '''
    context_groups = Typed(dict, {})
    context_items = Typed(dict, {})
    context_meta = Typed(dict, {})

    selectors = Typed(dict, ())
    symbols = Typed(dict, ())

    # Reflects state of selectors and context_items as currently applied.
    _context_item_state = Typed(dict, ())
    _selector_state = Typed(dict, ())

    _selectors = Typed(dict, ())

    changes_pending = Bool(False)

    _iterators = Typed(dict, ())
    _namespace = Typed(ExpressionNamespace, ())
    _prior_values = Typed(list, ())

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
        selectors = {}
        point = self.workbench.get_extension_point(SELECTORS_POINT)
        for extension in point.extensions:
            for selector in extension.get_children(BaseSelector):
                selectors[selector.name] = selector
        self.selectors = selectors

    def _refresh_symbols(self, event=None):
        symbols = {}
        point = self.workbench.get_extension_point(SYMBOLS_POINT)
        for extension in point.extensions:
            for symbol in extension.get_children(Symbol):
                symbols[symbol.name] = symbol.get_object() 
        self.symbols = symbols

    def _refresh_items(self, event=None):
        log.debug('Refreshing context items')
        context_groups = {}
        context_items = {}
        context_meta = {}
        items = []
        groups = []
        meta = []

        point = self.workbench.get_extension_point(ITEMS_POINT)
        for extension in point.extensions:
            items.extend(extension.get_children(ContextItem))
            groups.extend(extension.get_children(ContextGroup))
            meta.extend(extension.get_children(ContextMeta))

        for group in groups:
            log.debug('Adding context group {}'.format(group.name))
            if group.name in context_groups:
                m = 'Context group {} already defined'.format(group.name)
                raise ValueError(m)
            context_groups[group.name] = group
            for item in group.children:
                item.group = group.name
                items.append(item)

        for item in items:
            log.trace('Adding context item {}'.format(item.name))
            if item.group not in context_groups:
                m = 'Group {} for {} does not exist'
                m = m.format(item.group, item.name)
                raise ValueError(m)
            if item.name in context_items:
                m = 'Context item {} already defined'.format(item.name)
                raise ValueError(m)
            for m in meta:
                item.meta[m] = m.default_value

            context_items[item.name] = item

        self.context_items = context_items
        self.context_groups = context_groups

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

    def _observe_item_updated(self, event):
        self._check_for_changes()

    def _observe_item_rove(self, event):
        if event['value']:
            self.rove_item(event['object'])
        else:
            self.unrove_item(event['object'])

    @observe('selectors')
    def _bind_selectors(self, change):
        for p in change.get('oldvalue', {}).values():
            p.unobserve('updated', self._observe_selector_updated)
        for p in change.get('value', {}).values():
            p.observe('updated', self._observe_selector_updated)

    def _observe_selector_updated(self, event):
        self._check_for_changes()

    def _get_expressions(self):
        # Return a dictionary of expressions for all context_items that are not
        # managed by the selectors.
        return {k: c.expression for k, c in self.context_items.items() \
                if isinstance(c, Parameter) and not c.rove}

    def _get_iterators(self):
        return {k: v.get_iterator() for k, v in self.selectors.items()}

    def iter_settings(self, iterator, cycles=None):
        selector = self.selectors[iterator].get_iterator(cycles=cycles)
        for expressions in selector:
            self._namespace.reset()
            self._namespace.update_expressions(expressions)
            yield self.get_values()

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

    def unrove_item(self, item):
        for selector in self.selectors.values():
            if item in selector.context_items:
                selector.remove_item(item)

    def get_context_info(self):
        return dict((i, self.get_item_info(i)) for i in self.context_items)

    def next(self, save_prior, selector, results):
        '''
        Shortcut for advancing to the next setting.
        '''
        self.next_setting(save_prior)
        self.next_selector_setting(selector)
        self.set_values(results)

    def next_setting(self, selector=None, save_prior=True):
        '''
        Load next set of expressions. If there are no selectors defined, then
        this essentially clears the namespace and allows expresssions to be
        recomputed.
        '''
        if save_prior:
            prior_values = self._prior_values[:]
            prior_values.append(self.get_values())
            self._prior_values = prior_values
        self._namespace.reset()

        if selector is  None:
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

    def _check_for_changes(self):
        for name, state in self._context_item_state.items():
            if name not in self.context_items:
                self.changes_pending = True
                return
            item = self.context_items[name] 
            if (item.rove, item.expression) != state:
                self.changes_pending = True
                return
        self.changes_pending = self.selectors != self._selectors

    def apply_changes(self):
        self._apply_context_item_state()
        self._apply_selector_state()
        self._namespace.update_expressions(self._get_expressions())
        self._namespace.update_symbols(self.symbols)
        self._iterators = self._get_iterators()
        self.changes_pending = False

    def revert_changes(self):
        self._revert_context_item_state()
        self._revert_selector_state()
        self.changes_pending = False

    def _get_parameters(self):
        return {n: i for n, i in self.context_items.items() \
                if isinstance(i, Parameter)}

    def _get_expressions(self):
        return {n: i.expression for n, i in self._get_parameters().items() \
                if not i.rove}

    def _apply_selector_state(self):
        for name, selector in self.selectors.items():
            self._selector_state[name] = deepcopy(selector.__getstate__())

    def _revert_selector_state(self):
        for name, selector in self.selectors.items():
            selector.__setstate__(deepcopy(self._selector_state[name]))

    def _apply_context_item_state(self):
        for name, item in self.context_items.items():
            self._context_item_state[name] = deepcopy(item.__getstate__())

    def _revert_context_item_state(self):
        for name, item in self.context_items.items():
            item.__setstate__(deepcopy(self._context_item_state[name]))
