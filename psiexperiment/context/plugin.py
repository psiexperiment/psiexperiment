# TODO: Implement mechanism to access results of prior trial?
import cPickle as pickle
from copy import deepcopy

from atom.api import Typed, Bool, Str, observe
from enaml.application import deferred_call
from enaml.layout.api import InsertItem, InsertTab
from enaml.workbench.plugin import Plugin

from .context_item import ContextItem, Parameter
from .context_group import ContextGroup
from .selector import BaseSelector
from .expression import ExpressionNamespace

SELECTORS_POINT = 'psiexperiment.context.selectors'
ITEMS_POINT = 'psiexperiment.context.items'
#SYMBOLS_POINT = 'psiexperiment.context.symbols'


def copy_attrs(from_atom, to_atom):
    if to_atom.__class__ != from_atom.__class__:
        raise ValueError()
    for member in to_atom.members():
        try:
            value = deepcopy(getattr(from_atom, member))
            setattr(to_atom, member, value)
        except:
            pass


class ContextPlugin(Plugin):
    '''
    Plugin that provides a sequence of values that can be used by a controller
    to determine the experiment parameters.
    '''
    context_groups = Typed(dict, {})
    context_items = Typed(dict, {})

    selectors = Typed(dict, ())
    symbols = Typed(dict, ())

    # Reflects state of selectors and parameters as currently applied.
    _selectors = Typed(dict, ())
    _context_items = Typed(dict, ())

    changes_pending = Bool(False)

    _iterators = Typed(dict, ())
    _namespace = Typed(ExpressionNamespace, ())
    _prior_values = Typed(list, ())

    def start(self):
        self._refresh_selectors()
        self._refresh_items()
        self._bind_observers()
        try:
            # Attempt to load the default context settings. This may fail if we
            # have made changes to the code (i.e., added or removed parameters).
            core = self.workbench.get_plugin('enaml.workbench.core')
            core.invoke_command('psiexperiment.get_default_context')
        except:
            #raise
            pass

    def stop(self):
        self._unbind_observers()

    def _refresh_selectors(self):
        selectors = {}
        point = self.workbench.get_extension_point(SELECTORS_POINT)
        for extension in point.extensions:
            for selector in extension.get_children(BaseSelector):
                selectors[selector.name] = selector
        self.selectors = selectors

    def _refresh_items(self):
        context_groups = {}
        context_items = {}

        point = self.workbench.get_extension_point(ITEMS_POINT)
        for extension in point.extensions:
            items = extension.get_children(ContextItem)
            groups = extension.get_children(ContextGroup)
            for group in groups:
                if group.name in context_groups:
                    raise ValueError('Context group %s already defined',
                                     group.name)
                context_groups[group.name] = group
            for item in items:
                if item.group not in context_groups:
                    raise ValueError('Group %s for %s does not exist',
                                     item.group, item.name)
                if item.name in context_items:
                    raise ValueError('Context item %s already defined',
                                     item.name)
                context_items[item.name] = item
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
            if hasattr(i, 'rove'):
                i.unobserve('rove', self._observe_item_rove)
                if i.rove:
                    self.remove_parameter(i)
            if hasattr(i, 'expression'):
                i.unobserve('expression', self._observe_item_expression)
        for i in change.get('value', {}).values():
            if hasattr(i, 'rove'):
                i.observe('rove', self._observe_item_rove)
                if i.rove:
                    self.append_parameter(i)
            if hasattr(i, 'expression'):
                i.observe('expression', self._observe_item_expression)

    def _observe_item_expression(self, event):
        self._check_for_changes()

    def _observe_item_rove(self, event):
        parameter = event['object']
        if parameter.rove:
            self.append_parameter(parameter)
        else:
            self.remove_parameter(parameter)
        self._check_for_changes()

    @observe('selectors')
    def _bind_selectors(self, change):
        for p in change.get('oldvalue', {}).values():
            p.unobserve('updated', self._observe_selector_updated)
        for p in change.get('value', {}).values():
            p.observe('updated', self._observe_selector_updated)

    def _observe_selector_updated(self, event):
        self._check_for_changes()

    def _update_attrs(self, context_items, selectors):
        for i in self.context_items:
            from_items = context_items[i]
            to_items = self.context_items[i]
            copy_attrs(from_items, to_items)
        for s in self.selectors:
            from_selector = selectors[s]
            to_selector = self.selectors[s]
            copy_attrs(from_selector, to_selector)

    def _check_for_changes(self):
        context_items_changed = self.context_items != self._context_items
        selectors_changed = self.selectors != self._selectors
        self.changes_pending = context_items_changed or selectors_changed

    def _get_expressions(self):
        # Return a dictionary of expressions for all parameters that are not
        # managed by the selectors.
        expressions = {}
        for i in self.context_items.values():
            if isinstance(i, Parameter) and not getattr(i, 'rove', False):
                expressions[i.name] = i.expression
        return expressions

    def _get_sequences(self):
        return dict((n, s.__getstate__()) for n, s in self.selectors.items())

    def _get_iterators(self):
        return dict((k, v.get_iterator()) for k, v in self.selectors.items())

    def get_context_info(self):
        context_info = {}
        for i in self.context_items.values():
            info = dict(dtype=i.dtype,
                        label=i.label,
                        compact_label=i.compact_label,
                        rove=getattr(i, 'rove', False)
                        )
            context_info[i.name] = info
        return context_info

    def append_parameter(self, parameter):
        for selector in self.selectors.values():
            if parameter not in selector.parameters:
                selector.append_parameter(parameter)

    def remove_parameter(self, parameter):
        for selector in self.selectors.values():
            if parameter in selector.parameters:
                selector.remove_parameter(parameter)

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
                return self.context_name[context_name].default
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
        self._namespace.update_expressions(self._get_expressions())
        self._namespace.update_symbols(self.symbols)
        self._iterators = self._get_iterators()
        self.changes_pending = False

    def revert_changes(self):
        self._update_attrs(self._context_items, self._selectors)
        self._check_for_changes()
