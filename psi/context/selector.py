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
and selectors. To illustrate what happens under the hood, let's create two
|parameters|, level and frequency:

    >>> from .api import Parameter
    >>> frequency = Parameter(name='frequency')
    >>> level = Parameter(name='level')

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
'''

import numpy as np

import functools
import itertools
import operator

from atom.api import Typed, Enum, Event, Bool, Property
from enaml.core.api import d_
from psi.core.enaml.api import PSIContribution

from . import choice


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

    context_items = Typed(list, [])
    updated = Event()

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
        #value = item.coerce_to_type(value)
        self.setting[item.name] = value
        self.updated = True


class CartesianProduct(BaseSelector):
    '''
    Generate all possible permutations of the values. The order in which the
    context items were added to the selector define the order in which the
    values are looped through (first item is slowest-varying and last item is
    fastest-varying).  The followign code illustrates the concept where item A
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
        super(SequenceSelector, self).append_item(item)

    def sort_settings(self):
        key = lambda x: [x[i.name] for i in self.context_items]
        settings = self.settings.copy()
        settings.sort(key=key)
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
        key = operator.itemgetter(*self.context_items) \
            if self.context_items else None
        selector = choice.options[self.order]
        return selector(settings, cycles, key=key)

    def set_value(self, setting_index, item, value):
        # TODO: It's weird that some methods take the index of the setting,
        # while others take the setting object. Need to sanitize this.
        #value = item.coerce_to_type(value)
        self.settings[setting_index][item.name] = value
        self.updated = True

    def get_value(self, setting_index, item):
        return self.settings[setting_index][item.name]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
