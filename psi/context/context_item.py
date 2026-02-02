import logging
log = logging.getLogger(__name__)

import numpy as np

from enaml.core.declarative import Declarative, d_
from atom.api import (set_default, Str, Typed, Value, Enum, Int, List, Event,
                      Property, observe, Bool, Dict, Coerced)

from psi.core.enaml.api import PSIContribution
from ..util import get_tagged_members, get_tagged_values


################################################################################
# ContextMeta
################################################################################
class ContextMeta(Declarative):

    name = d_(Str())
    label = d_(Str())
    link_rove = d_(Bool(True))
    editable = d_(Bool(False))


class UnorderedContextMeta(ContextMeta):

    values = d_(Coerced(set))

    def add_item(self, item):
        if item not in self.values:
            values = self.values.copy()
            values.add(item)
            self.values = values

    def remove_item(self, item):
        if item in self.values:
            values = self.values.copy()
            values.remove(item)
            self.values = values


class OrderedContextMeta(ContextMeta):

    values = d_(List())

    #: List of context items to be included.
    mandatory_items = d_(List())

    #: List of context items that can never be selected for this.
    forbidden_items = d_(List())

    def _default_values(self):
        return self.mandatory_items.copy()

    def _observe_values(self, event):
        # Make sure that all mandatory items are in the list and forbidden
        # items not in the list.
        values = self.values.copy()
        update = False
        for item in self.mandatory_items:
            if item not in self.values:
                values.append(item)
                update = True
        for item in self.forbidden_items:
            if item in self.values:
                values.remove(item)
                update = True
        if update:
            self.values = values

    def set_choice(self, choice, context_item):
        if context_item in self.forbidden_items:
            return
        if context_item in self.mandatory_items and choice is None:
            return

        values = self.values[:]
        if choice is None:
            values.remove(context_item)
        else:
            position = int(choice)-1
            if context_item in values:
                values.remove(context_item)
            values.insert(position, context_item)
        self.values = values

    def get_choice(self, context_item):
        try:
            return str(self.values.index(context_item) + 1)
        except ValueError:
            return None

    def get_choices(self, context_item):
        n = len(self.values)
        if context_item in self.forbidden_items:
            return []
        if context_item not in self.values:
            n += 1
        if context_item in self.mandatory_items:
            # Item must be in list somewhere. Don't have an entry that allows
            # the user to remove the item. But, user can rearrange items in
            # this list.
            return [str(i+1) for i in range(n)]
        else:
            return [None] + [str(i+1) for i in range(n)]


################################################################################
# Expression
################################################################################
class Expression(Declarative):

    #: Parameter that is assigned the result of the expression
    parameter = d_(Str())

    #: Expression to be evaluated
    expression = d_(Str())


################################################################################
# ContextGroup and ContextSet
################################################################################
class ContextGroup(PSIContribution):
    '''
    Used to group together context items in a single dock item pane.
    '''
    #: Are the parameters in this group visible?
    visible = d_(Bool())
    hide_when = d_(Enum('empty', 'never', 'always'))

    def _update_visible(self):
        visible = False
        if self.hide_when == 'never':
            visible = True
        for c in self.subgroups:
            if c.visible:
                visible = True
        self.visible = visible

    #: Fired whenever the children change.
    updated = d_(Event())

    subgroups = Property()

    def _get_subgroups(self):
        classes = (ContextItem, ContextSet)
        return [c for c in self.children if isinstance(c, classes)]


class ContextSet(PSIContribution):
    '''
    Used to group together context items for special formatting.
    '''
    fmt = d_(List())

    #: Name of the group to display the item under. This should never be
    #: overwitten even if we remove the item from the group (e.g., when
    #: loading/unloading plugin tokens).
    group_name = d_(Str())

    #: Are the parameters in this set visible?
    visible = d_(Bool(True))


class ContextRow(ContextSet):
    '''
    Used to group together context items into a single row that is formatted.
    '''
    pass


################################################################################
# ContextItem
################################################################################
class ContextItem(Declarative):
    '''
    Defines the core elements of a context item. These items are made available
    to the context namespace.
    '''
    #: Must be a valid Python identifier. Used by eval() in expressions.
    name = d_(Str())

    #: Long-format label for display in the GUI. Include units were applicable.
    label = d_(Str())

    # Datatype of the value. Required for properly initializing some data
    # plugins (e.g., those that save data to a HDF5 file).
    dtype = d_(Str())

    #: Name of the group to display the item under. This should never be
    #: overwitten even if we remove the item from the group (e.g., when
    #: loading/unloading plugin tokens).
    group_name = d_(Str())

    # Compact label where there is less space in the GUI (e.g., under a column
    # heading for example).
    compact_label = d_(Str())

    # Is this visible via the standard configuration menus?
    visible = d_(Bool(True))

    # Can this be configured by the user? This will typically be False if the
    # experiment configuration has contributed an Expression that assigns the
    # value of this parameter.
    editable = Bool(True)

    updated = Event()

    def _default_label(self):
        return self.name.capitalize().replace('_', ' ')

    def _default_compact_label(self):
        return self.label

    def coerce_to_type(self, value):
        coerce_function = np.dtype(self.dtype).type
        value = coerce_function(value)
        return value.item()

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self}>'

    def get_flags(self):
        flags = []
        if isinstance(self.parent, ContextGroup):
            flags.append(f'aaagroup {self.parent.name}')
        elif self.group_name:
            flags.append(f'group {self.group_name}')
        if not self.visible:
            flags.append('not visible')
        if not self.editable:
            flags.append('not editable')
        return flags

    def __str__(self):
        flags = ', '.join(self.get_flags())
        return f'{self.name} ({flags})'

    def get_preferences(self):
        return get_tagged_values(self, 'preference')

    def set_preferences(self, preferences):
        for member in get_tagged_members(self, 'preference'):
            if member in preferences:
                try:
                    setattr(self, member, preferences.get(member))
                except AttributeError:
                    m = f'Unable to restore saved value for {member} of {self.name} ({self})'
                    log.error(m)


class Result(ContextItem):
    '''
    A context item whose value is set by the plugin that contributes it. Typical
    use-cases include results calculated (e.g., reaction time, hit vs miss)
    following completion of a trial. These values are made available to the next
    iteration of the context.
    '''
    pass


class Parameter(ContextItem):
    '''
    A context item that can be evaluated dynamically, but cannot be included as
    part of a selector.  This is typically used for settings that must be
    determined before values are drawn from the selectors (e.g., probability of
    a go trial).
    '''
    # Default value of the context item when used as part of a selector.
    default = d_(Value())

    expression = d_(Str()).tag(preference=True)

    # Defines the span over which the item's value does not change:
    # * experiment - the value cannot change once the experiment begins
    # * trial - The value cannot change once a trial begins. This is the only
    #   type of item that can be roved using a selector.
    # * arbitrary - The value can be changd at any time but it does not make
    #   sense for it to be a roving item.
    scope = d_(Enum('trial', 'experiment', 'arbitrary'))

    # Is the value of this item managed by a selector?
    rove = d_(Bool()).tag(preference=True)

    def _default_expression(self):
        return str(self.default)

    def _default_dtype(self):
        return np.array(self.default).dtype.str

    def _default_label(self):
        return self.name

    def to_expression(self, value):
        return str(value)

    def set_value(self, value):
        self.expression = self.to_expression(value)

    def get_flags(self):
        flags = super().get_flags()
        flags.append(f'scope {self.scope}')
        return flags


class EnumParameter(Parameter):

    expression = Property().tag(transient=True)
    selected = d_(Str()).tag(preference=True)
    choices = d_(Dict())
    default = d_(Str())

    def _default_dtype(self):
        values = list(self.choices.values())
        return np.array(values).dtype.str

    def _get_expression(self):
        return self.choices.get(self.selected, None)

    def _set_expression(self, expression):
        for k, v in self.choices.items():
            if v == expression:
                self.selected = k
                break
        else:
            if expression is not None:
                valid_choices = ', '.join(self.choices.values())
                t = 'Could not map expression {} to choice. Valid choices are {}.'
                raise ValueError(t.format(expression, valid_choices))

    def _default_selected(self):
        if self.default not in self.choices:
            return next(iter(self.choices))
        return self.default

    @observe('selected')
    def _notify_update(self, event):
        if self.is_initialized:
            self.notify('expression', self.expression)

    def to_expression(self, value):
        return str(self.choices.get(value, None))

    def coerce_to_type(self, value):
        return str(value)


class MultiSelectParameter(Parameter):
    '''
    Builds a list of selected values from a set of choices
    '''
    expression = Property().tag(transient=True)
    selected = d_(List()).tag(preference=True)
    choices = d_(Dict())
    default = d_(List())

    #: Number of columns to show for choices before starting a new row.
    n_cols = d_(Int(3))
    button_width = d_(Int(40))
    quote_values = d_(Bool(False))
    min_selected = d_(Int(1))

    def _default_dtype(self):
        values = list(self.choices.values())
        return np.array(values).dtype.str

    def _get_expression(self):
        # Sort to ensure that expression is always rendered the same regardless
        # of order of selection.
        try:
            choices = sorted([self.choices[s] for s in self.selected])
        except KeyError:
            valid_keys = ', '.join(self.choices.keys())
            raise KeyError('Invalid key. Valid keys are %s.' % valid_keys)
        if self.quote_values:
            choices = [f"'{c}'" for c in choices]
        else:
            choices = [f"{c}" for c in choices]
        return f'[{", ".join(choices)}]'

    def _set_expression(self, expression):
        choices = eval(expression)
        #choices = [eval(c.strip()) for c in expression.strip('[]').split(', ')]
        #if self.quote_values:
            #choices = [c.strip("'") for c in choices]
        selected = []
        for k, v in self.choices.items():
            if v in choices:
                selected.append(k)
        self.selected = selected

    def _default_selected(self):
        return self.default

    @observe('selected')
    def _notify_update(self, event):
        if self.is_initialized:
            log.error(self.selected)
            self.notify('expression', self.expression)

    def to_expression(self, value):
        return str(self.choices.get(value, None))

    def coerce_to_type(self, value):
        return str(value)


class FileParameter(Parameter):

    expression = Property().tag(transient=True)
    path = d_(Str())
    file_mode = d_(Enum('any_file', 'existing_file', 'directory'))
    current_path = d_(Str())
    name_filters = d_(List(Str()))
    dtype = set_default('U')

    def _get_expression(self):
        return '"{}"'.format(self.path)

    def _set_expression(self, expression):
        self.path = expression.strip('\"\'')

    @observe('path')
    def _notify_update(self, event):
        self.updated = event


class BoolParameter(Parameter):

    dtype = bool
