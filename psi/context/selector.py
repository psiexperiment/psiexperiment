import numpy as np

import functools
import itertools
import operator
import collections
from copy import deepcopy

from atom.api import Atom, ContainerList, Typed, Enum, Event, Bool, Property, Float
from enaml.core.declarative import Declarative, d_
from psi.core.enaml.api import PSIContribution

from . import choice
from .. import SimpleState


class BaseSelector(PSIContribution):

    context_items = Typed(list, [])
    updated = Event()

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
        context_items = self.context_items[:]
        context_items.append(item)
        self.context_items = context_items
        self.updated = True

    def remove_item(self, item):
        context_items = self.context_items[:]
        context_items.remove(item)
        self.context_items = context_items
        self.updated = True


class SingleSetting(BaseSelector):

    setting = Typed(dict, ()).tag(preference=True)

    def append_item(self, item):
        if item.name not in self.setting:
            self.setting[item.name] = item.default
        super(SingleSetting, self).append_item(item)

    def get_iterator(self, cycles=None):
        setting = {i: self.setting[i.name] for i in self.context_items}
        if cycles is None:
            return itertools.cycle([setting])
        else:
            return [setting]*cycle

    def get_value(self, item):
        return self.setting[item.name]

    def set_value(self, item, value):
        self.setting[item.name] = value
        self.updated = True


class CartesianProduct(BaseSelector):

    settings = Typed(dict, {}).tag(preference=True)

    def append_item(self, item):
        self.settings.setdefault(item.name, [])
        super().append_item(item)

    def add_setting(self, item, value):
        self.settings[item.name].append(value)

    def get_settings(self):
        values = [self.settings[i.name] for i in self.context_items]
        return [dict(zip(self.context_items, s)) for s in itertools.product(*values)]

    def get_iterator(self, cycles=np.inf):
        settings = self.get_settings()
        return choice.exact_order(settings, cycles)


class SequenceSelector(BaseSelector):

    settings = Typed(list, []).tag(preference=True)
    order = d_(Enum(*choice.options.keys())).tag(preference=True)

    def add_setting(self, values=None):
        if values is None:
            values = {}
        for item in self.context_items:
            if item.name not in values:
                values[item.name] = item.default
        settings = self.settings[:]
        settings.append(values)
        self.settings = settings
        self.updated = True

    def remove_setting(self, setting):
        settings = self.settings[:]
        settings.remove(setting)
        self.settings = settings
        self.updated = True

    def append_item(self, item):
        for setting in self.settings:
            if item not in setting:
                setting[item.name] = item.default
        super(SequenceSelector, self).append_item(item)

    def sort_settings(self):
        self.settings.sort()
        self.updated = True

    def _observe_order(self, event):
        self.updated = True

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
        self.settings[setting_index][item.name] = value
        self.updated = True

    def get_value(self, setting_index, item):
        return self.settings[setting_index][item.name]
