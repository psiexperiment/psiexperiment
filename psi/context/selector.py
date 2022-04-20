'''
Introduction
------------
This module contains a collection of selectors that facilitate generating a
sequence of settings for an experiment. A selector can manage multiple context
items. On each iteration, the selector is responsible for returning a
dictionary of settings. Dictionary keys are the context items and values are
the value for that context item.

Examples
--------
The plugins typically handle most of the configuration for the context items
and selectors. These are not typically meant for direct use. To illustrate what
happens under the hood, let's create two |parameters|, level and frequency:

    >>> from psi.context.api import Parameter
    >>> frequency = Parameter(name='frequency', dtype='float64')
    >>> level = Parameter(name='level', dtype='float64')

Now, let's create the selector and add the parameters to the selector:

    >>> selector = CartesianProduct()
    >>> selector.append_item(frequency)
    >>> selector.append_item(level)
    >>> for i in range(20, 81, 20):
    ...     selector.add_setting(level, i)
    >>> selector.add_setting(frequency, 4000)
    >>> selector.add_setting(frequency, 8000)

Finally, create the iterator that will generate the sequence of values and loop
through it. We have specified that it only runs through one *cycle*. For the
`CartesianProduct` selector, this means once the cartesian product of all
possible combinations of the parameters have been generated, it will raise a
`StopIteration`. Since there are 8 possible combinations of two frequencies,
and four levels, let's see what happens first:

    >>> iterator = selector.get_iterator(cycles=1)
    >>> for i in range(8):
    ...     print(next(iterator))
    {<Parameter: frequency>: 4000, <Parameter: level>: 20}
    {<Parameter: frequency>: 4000, <Parameter: level>: 40}
    {<Parameter: frequency>: 4000, <Parameter: level>: 60}
    {<Parameter: frequency>: 4000, <Parameter: level>: 80}
    {<Parameter: frequency>: 8000, <Parameter: level>: 20}
    {<Parameter: frequency>: 8000, <Parameter: level>: 40}
    {<Parameter: frequency>: 8000, <Parameter: level>: 60}
    {<Parameter: frequency>: 8000, <Parameter: level>: 80}

Now, what happens if we continue past these 8?

    >>> next(iterator)
    Traceback (most recent call last):
     ...
    StopIteration

The selectors can accept a mix of values and expressions. Expressions are
returned intact:

    >>> selector = SequenceSelector(order='ascending')
    >>> selector.append_item(frequency)
    >>> selector.add_setting({'frequency': '4000 * 1.2'})
    >>> selector.add_setting({'frequency': '2000 * 1.2'})
    >>> selector.add_setting({'frequency': '8000 * 1.2'})
    >>> selector.add_setting({'frequency': 4000})
    >>> selector.add_setting({'frequency': 2000})
    >>> selector.add_setting({'frequency': 8000})
    >>> iterator = selector.get_iterator(cycles=1)
    >>> for i in range(6):
    ...     print(next(iterator))
    {<Parameter: frequency>: 2000}
    {<Parameter: frequency>: '2000 * 1.2'}
    {<Parameter: frequency>: 4000}
    {<Parameter: frequency>: '4000 * 1.2'}
    {<Parameter: frequency>: 8000}
    {<Parameter: frequency>: '8000 * 1.2'}

The context plugin can provide a namespace that is used to evaluate the
expression. To emulate this behavior for an AM depth sequence where you want to
express values in dB re 100% but have the values internally converted to 0 ...
1 since the SAM token generation does not accept dB-scaled values. This will
give you a nicely-sorted list:

    >>> selector = SequenceSelector(order='descending')
    >>> am_depth = Parameter(name='am_depth', dtype='float64')
    >>> selector.append_item(am_depth)
    >>> selector.symbols = {'dbi': lambda x: 10**(x/20)}

    >>> selector.add_setting({'am_depth': 'dbi(-3)'})
    >>> selector.add_setting({'am_depth': 'dbi(-12)'})
    >>> selector.add_setting({'am_depth': 'dbi(-9)'})
    >>> selector.add_setting({'am_depth': 'dbi(0)'})
    >>> selector.add_setting({'am_depth': 'dbi(-6)'})
    >>> iterator = selector.get_iterator(cycles=1)
    >>> for i in range(5):
    ...     print(next(iterator))
    {<Parameter: am_depth>: 'dbi(0)'}
    {<Parameter: am_depth>: 'dbi(-3)'}
    {<Parameter: am_depth>: 'dbi(-6)'}
    {<Parameter: am_depth>: 'dbi(-9)'}
    {<Parameter: am_depth>: 'dbi(-12)'}

'''
import logging
log = logging.getLogger(__name__)

import numpy as np

import functools
import itertools
import operator

from atom.api import Bool, Dict, Enum, Event, Property, set_default, Typed
from enaml.core.api import d_
from psi.core.enaml.api import PSIContribution

from psi.context import choice


def warn_empty(method):
    '''
    Used to intercept potentially confusing error messages and translate them
    to a more user-friendly error message that indicates that some values may
    not have been specified for all context items.
    '''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except ValueError as e:
            mesg = f'{self.label} sequence must have at least one value'
            raise ValueError(mesg) from e
    return wrapper


class BaseSelector(PSIContribution):
    '''
    Defines a selector where items can be added/removed
    '''
    symbols = Typed(dict, {})
    updated = Event()

    def _default_name(self):
        return 'default'

    #: Can the user manage the selector by manually selecting items to rove in
    #: the GUI? If False, the experiment paradigm needs to handle the selection
    #: of the roving parameters.
    user_managed = d_(Bool(True))

    context_items = Typed(list, [])

    #: Since order of context items is important for certain selectors (e.g.,
    #: the CartesianProduct), this attribute is used to persist experiment
    #: settings when saving/loading from a file.
    context_item_order = Property().tag(preference=True)

    def _get_context_item_order(self):
        return [i.name for i in self.context_items]

    def _set_context_item_order(self, order):
        old_items = self.context_items[:]
        new_items = []
        for name in order:
            for item in old_items[:]:
                if item.name == name:
                    new_items.append(item)
                    old_items.remove(item)

        # Be sure to tack on any old items that were not saved to the ordering
        # in the preferences file.
        new_items.extend(old_items)
        self.context_items = new_items

    def append_item(self, item):
        '''
        Add context item to selector

        Parameters
        ----------
        item : ContextItem
            Item to add to selector
        '''
        context_items = self.context_items[:]
        context_items.append(item)
        self.context_items = context_items
        self.updated = True

    def remove_item(self, item):
        '''
        Remove context item from selector

        Parameters
        ----------
        item : ContextItem
            Item to remove from selector
        '''
        context_items = self.context_items[:]
        context_items.remove(item)
        self.context_items = context_items
        self.updated = True

    def find_item(self, name):
        for item in self.context_items:
            if item.name == name:
                return item
        raise ValueError(f'{name} not in selector {self.name}')


class SingleSetting(BaseSelector):
    '''
    Each parameter takes on only a single value. The value is the same on every
    iteration. This is commonly used in behavioral paradigms where you want to
    alternate between two selectors. For exampe, in a go-nogo task you may have
    multiple settings you want to test for the go and only one you want to test
    for the nogo. You would use a `SequenceSetting` selector for the go and
    `SingleSetting` selector for the nogo.
    '''
    setting = Typed(dict, ()).tag(preference=True)

    def append_item(self, item):
        '''
        Add context item to selector

        Parameters
        ----------
        item : ContextItem
            Item to add to selector
        '''
        if item.name not in self.setting:
            self.setting[item.name] = item.default
        super(SingleSetting, self).append_item(item)

    @warn_empty
    def get_iterator(self, cycles=None):
        '''
        Returns iterator that produces a dictionary containing settings for the
        managed context items on each iteration.

        Parameters
        ----------
        cycles : {None, int}
            Number of cycles before raising StopIteration. If None, continue
            forever. Here, cycle means the number of times the iterator loops
            through the full sequence of settings defined in the selector. The
            number of settings on eacy cycle depends on the selector. For
            example, the CartesianProduct selector will produce all possible
            permutations of parameters on each cycle.
        '''
        setting = {i: self.setting[i.name] for i in self.context_items}
        if cycles is None:
            return itertools.cycle([setting])
        else:
            return [setting]*cycle

    def get_value(self, item):
        '''
        Get the value for the item

        Parameters
        ----------
        item : ContextItem
            Item to get value for
        '''
        return self.setting[item.name]

    def set_value(self, item, value):
        '''
        Set the value for the item

        Parameters
        ----------
        item : ContextItem
            Item to set value for
        '''
        self.setting[item.name] = value
        self.updated = True


class CartesianProduct(BaseSelector):
    '''
    Generate all possible permutations of the values. The order in which the
    context items were added to the selector define the order in which the
    values are looped through (first item is slowest-varying and last item is
    fastest-varying).  The following code illustrates the concept where item A
    was added first, followed by item B and then item C:

        for value_A in item_A_values:
            for value_B in item_B_values:
                for value_C in item_C_values:
                    ...
    '''
    settings = Typed(dict, {}).tag(preference=True)

    def append_item(self, item):
        '''
        Add context item to selector

        Parameters
        ----------
        item : ContextItem
            Item to add to selector
        '''
        self.settings.setdefault(item.name, [])
        super().append_item(item)

    def add_setting(self, item, value):
        self.settings[item.name].append(value)

    def get_settings(self):
        values = [self.settings[i.name] for i in self.context_items]
        return [dict(zip(self.context_items, s)) for s in itertools.product(*values)]

    @warn_empty
    def get_iterator(self, cycles=np.inf):
        settings = self.get_settings()
        return choice.exact_order(settings, cycles)


class SequenceSelector(BaseSelector):
    '''
    TODO
    '''

    settings = Typed(list, []).tag(preference=True)
    order = d_(Enum(*choice.options.keys())).tag(preference=True)

    def add_setting(self, values=None, index=None):
        if values is None:
            values = {}
        for item in self.context_items:
            if item.name not in values:
                values[item.name] = item.default
        settings = self.settings[:]
        if index is None:
            settings.append(values)
        else:
            settings.insert(index, values)
        self.settings = settings
        self.updated = True

    def remove_setting(self, setting):
        settings = self.settings[:]
        settings.remove(setting)
        self.settings = settings
        self.updated = True

    def append_item(self, item):
        '''
        Add context item to selector

        Parameters
        ----------
        item : ContextItem
            Item to add to selector
        '''
        for setting in self.settings:
            if item not in setting:
                setting[item.name] = item.default
        super().append_item(item)

    def get_key(self, settings, use='name'):
        key = []
        for item in self.context_items:
            if use == 'name':
                setting_value = settings[item.name]
            elif use == 'item':
                setting_value = settings[item]
            else:
                raise ValueError(f'Unrecognized value for use: "{use}"')
            try:
                value = item.coerce_to_type(setting_value)
            except ValueError:
                value = item.coerce_to_type(eval(setting_value, self.symbols))
            key.append(value)
        return key

    def sort_settings(self):
        settings = self.settings.copy()
        settings.sort(key=self.get_key)
        self.settings = settings
        self.updated = True

    def _observe_order(self, event):
        self.updated = True

    @warn_empty
    def get_iterator(self, cycles=np.inf):
        # Some selectors need to sort the settings. To make sure that the
        # selector sorts the parameters in the order the columns are specified,
        # we need to convert to a list of tuples.
        settings = [{i: s[i.name] for i in self.context_items} \
                    for s in self.settings]
        selector = choice.options[self.order]
        return selector(settings, cycles, key=lambda x: self.get_key(x, 'item'))

    def set_value(self, setting_index, item, value):
        # TODO: It's weird that some methods take the index of the setting,
        # while others take the setting object. Need to sanitize this.
        self.settings[setting_index][item.name] = value
        self.updated = True

    def get_value(self, setting_index, item):
        return self.settings[setting_index][item.name]


class FriendlyCartesianProduct(BaseSelector):
    '''
    Like the CartesianProduct selector, but offers a much more user-friendly
    interface that allows for customizing the range and settings used.
    '''
    user_managed = set_default(True)

    #: Tracks the user-configurable values (start, end, step, and ordering)
    context_settings = Dict().tag(preference=True)

    #: Programatically-set details (e.g., transforms, spacing, etc.)
    context_detail = d_(Dict())

    #: If empty, all context items are selectable. Otherwise, list context
    #: items that can be selected for roving.
    can_manage = d_(Typed(list, []))

    def append_item(self, item):
        detail = self.context_detail.get(item.name, {})
        inverse_fn = detail.get('inverse_transform_fn', lambda x: x)

        settings = self.context_settings.get(item.name, {})
        settings.setdefault('start', item.default)
        settings.setdefault('end', item.default)
        settings.setdefault('start', inverse_fn(item.default))
        settings.setdefault('end', inverse_fn(item.default))
        settings.setdefault('step', 1)
        self.context_settings[item.name] = settings

        super().append_item(item)

    def get_values(self, item, transform=False):
        settings = self.context_settings[item.name]
        detail = self.context_detail[item.name]
        range_fn = detail.get('range_fn', lambda lb, ub, s: np.arange(lb, ub + s/2, s))
        values = range_fn(settings['start'], settings['end'], settings['step'])
        if transform:
            transform_fn = detail.get('transform_fn', lambda x: x)
            values = [transform_fn(v) for v in values]
        return values

    def get_settings(self):
        values = [self.get_values(item, True) for item in self.context_items]
        return [dict(zip(self.context_items, s)) for s in itertools.product(*values)]

    @warn_empty
    def get_iterator(self, cycles=np.inf):
        return choice.exact_order(self.get_settings(), cycles)

    def move_item_to(self, item_name, to_item_name):
        item_names = [i.name for i in self.context_items]
        a = item_names.index(item_name)
        b = item_names.index(to_item_name)
        context_items = self.context_items[:]
        context_items.insert(b, context_items.pop(a))
        self.context_items = context_items


if __name__ == '__main__':
    import doctest
    doctest.testmod()
