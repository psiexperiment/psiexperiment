import logging
log = logging.getLogger(__name__)

import numpy as np

from enaml.core.declarative import Declarative, d_
from atom.api import (Unicode, Typed, Value, Enum, List, Event, Property,
                      observe, Bool, Dict, Coerced)

from psi.core.enaml.api import PSIContribution


################################################################################
# ContextMeta
################################################################################
class ContextMeta(Declarative):

    name = d_(Unicode())
    label = d_(Unicode())
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

    def add_item(self, item):
        if item not in self.values:
            values = self.values.copy()
            values.append(item)
            self.values = values

    def remove_item(self, item):
        if item in self.values:
            values = self.values.copy()
            values.remove(item)
            self.values = values

    def _default_values(self):
        return []

    # TODO: move most of this stuff to the enaml interface
    def set_choice(self, choice, context_item):
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
        if context_item not in self.values:
            n += 1
        return [str(i+1) for i in range(n)]


################################################################################
# Expression
################################################################################
class Expression(Declarative):

    # Parameter that is assigned the result of the expression
    parameter = d_(Unicode())

    # Expression to be evaluated
    expression = d_(Unicode())


################################################################################
# ContextGroup
################################################################################
class ContextGroup(PSIContribution):
    '''
    Used to group together context items for management.
    '''
    # Group name
    name = d_(Unicode())

    # Label to use in the GUI
    label = d_(Unicode())

    # Are the parameters in this group visible?
    visible = d_(Bool(True))

    # Items in context
    items = List()

    has_visible_items = d_(Bool(False))

    def _check_visible(self):
        self.has_visible_items = bool(self.visible_items())

    def visible_items(self):
        if not self.visible:
            return []
        return [i for i in self.items if i.visible]

    def add_item(self, item):
        if item not in self.items:
            self.items = self.items[:] + [item]
            self._check_visible()
        else:
            raise ValueError(f'Item {item.name} already in group')

    def remove_item(self, item):
        if item in self.items:
            items = self.items[:]
            items.remove(item)
            self.items = items
            self._check_visible()


################################################################################
# ContextItem
################################################################################
class ContextItem(Declarative):
    '''
    Defines the core elements of a context item. These items are made available
    to the context namespace.
    '''
    # Must be a valid Python identifier. Used by eval() in expressions.
    name = d_(Unicode())

    # Long-format label for display in the GUI. Include units were applicable.
    label = d_(Unicode()).tag(preference=True)

    # Datatype of the value. Required for properly initializing some data
    # plugins (e.g., those that save data to a HDF5 file).
    dtype = d_(Unicode())

    group = d_(Typed(ContextGroup))

    # Name of the group to display the item under.
    group_name = d_(Unicode())

    # Compact label where there is less space in the GUI (e.g., under a column
    # heading for example).
    compact_label = d_(Unicode()).tag(preference=True)

    # Is this visible via the standard configuration menus?
    visible = d_(Bool(True)).tag(preference=True)

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

    def __str__(self):
        if self.group:
            return f'{self.name} in {self.group}'
        return f'{self.name}'

    def set_group(self, group):
        if self.group is not None and self.group != group:
            self.group.remove_item(self)

        self.group = group
        if self.group is not None:
            self.group.add_item(self)
            self.group_name = self.group.name
        else:
            self.group_name = ''


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

    expression = d_(Unicode()).tag(preference=True)

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


class EnumParameter(Parameter):

    expression = Property().tag(transient=True)
    selected = d_(Unicode()).tag(preference=True)
    choices = d_(Typed(dict))
    default = d_(Unicode())

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
                m = 'Could not map expression {} to choice'.format(expression)
                raise ValueError(m)

    def _default_selected(self):
        if self.default not in self.choices:
            return next(iter(self.choices))
        return self.default

    @observe('selected')
    def _notify_update(self, event):
        self.notify('expression', self.expression)

    def to_expression(self, value):
        return str(self.choices.get(value, None))

    def coerce_to_type(self, value):
        return str(value)


class FileParameter(Parameter):

    expression = Property().tag(transient=True)
    path = d_(Unicode())
    file_mode = d_(Enum('any_file', 'existing_file', 'directory'))
    current_path = d_(Unicode())
    name_filters = d_(List(Unicode()))

    def _get_expression(self):
        return '"{}"'.format(self.path)

    def _set_expression(self, expression):
        self.path = expression.strip('\"\'')

    @observe('path')
    def _notify_update(self, event):
        self.updated = event


class BoolParameter(Parameter):

    dtype = np.bool
