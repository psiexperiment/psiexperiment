# TODO:
# * Figure out how to add parameters that can only be evaluated once at the
#   beginning of an experiment (e.g., for setting sampling rate, etc.). One way
#   to do this would be to simply set a scope. By default, most parameters would
#   fall in the `trial` scope (i.e., can be changed from trial to trial).
#   However, experiment scope parameters would be frozen once the experiment
#   begins. Do we also want to implement a block scope? Some experiments revolve
#   around the concept of a trial block.
# * Add in some sanity checks. For example, if we define `go_probability`, we do
#   not want it to have an equation that depends on a rovable parameter. Is this
#   worthwhile to implement?

from enaml.core.declarative import Declarative, d_
from atom.api import Unicode, Typed, Bool, Value, Enum

from .. import SimpleState


class ContextItem(SimpleState, Declarative):
    '''
    Defines the core elements of a context item. These items are made available
    to the namespace for evaluation.
    '''

    # Must be a valid Python identifier. Used by eval() in expressions.
    name = d_(Unicode())

    # Long-format label for display in the GUI. Include units were applicable.
    label = d_(Unicode())

    # Datatype of the value. Required for properly initializing some data
    # plugins (e.g., those that save data to a HDF5 file).
    dtype = d_(Typed(type))

    # Name of the group to display the item under.
    group = d_(Unicode())

    # Compact label where there is less space in the GUI (e.g., under a column
    # heading for example).
    compact_label = d_(Unicode())

    # Attributes to compare to determine equality of two items.
    _cmp_attrs = ['name']

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for attr in self._cmp_attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


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

    # Expression used to determine value of item.
    expression = d_(Unicode())

    # Defines the span over which the item's value does not change:
    # * experiment - the value cannot change once the experiment begins
    # * trial - The value cannot change once a trial begins. This is the only
    #   type of item that can be roved using a selector.
    # * arbitrary - The value can be changd at any time but it does not make
    #   sense for it to be a roving item.
    scope = d_(Enum('experiment', 'trial', 'arbitrary'))

    _cmp_attrs = ContextItem._cmp_attrs + ['expression']
