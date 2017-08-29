import numpy as np

from enaml.core.declarative import Declarative, d_
from atom.api import (Unicode, Typed, Value, Enum, List, Event, Property,
                      observe, Bool)


class ContextMeta(Declarative):

    name = d_(Unicode())
    label = d_(Unicode())


class OrderedContextMeta(ContextMeta):

    values = d_(Typed(list, ()))
    default_value = None

    def set_value(self, position, context_item):
        values = self.values[:]
        if position is None:
            values.remove(context_item)
        else:
            if context_item in values:
                values.remove(context_item)
            values.insert(position, context_item)
        self.values = values

    def get_index(self, context_item):
        try:
            return self.values.index(context_item)
        except ValueError:
            return None

    def get_valid_indices(self, context_item):
        n = len(self.values)
        if context_item not in self.values:
            n += 1
        return list(range(n))


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

    # Name of the group to display the item under.
    group = d_(Unicode())

    # Compact label where there is less space in the GUI (e.g., under a column
    # heading for example).
    compact_label = d_(Unicode())

    updated = Event()

    def _default_label(self):
        return self.name.capitalize().replace('_', ' ')

    def _default_compact_label(self):
        return self.label

    def coerce_to_type(self, value):
        coerce_function = np.dtype(self.dtype).type
        value = coerce_function(value)
        return np.asscalar(value)


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
    default = d_(Value()).tag(preference=True)

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

    def to_expression(self, value):
        return str(value)


class EnumParameter(Parameter):

    expression = Property().tag(transient=True)
    choices = d_(Typed(dict))
    selected = d_(Unicode())
    default = d_(Unicode())

    def _default_dtype(self):
        return np.array(self.choices.values()).dtype.str

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
