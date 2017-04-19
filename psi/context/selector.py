import numpy as np

import functools
import itertools
import operator
import collections
from copy import deepcopy

from atom.api import ContainerList, Typed, Enum, Event, Bool
from enaml.core.declarative import Declarative, d_

from . import choice
from .. import SimpleState


class BaseSelector(SimpleState, Declarative):

    context_items = Typed(list, []).tag(transient=True)
    updated = Event().tag(transient=True)

    def append_item(self, item):
        context_items = self.context_items[:]
        context_items.append(item)
        print('appending', id(item))
        self.context_items = context_items
        self.updated = True

    def remove_item(self, item):
        context_items = self.context_items[:]
        context_items.remove(item)
        self.context_items = context_items
        self.updated = True


class SingleSetting(BaseSelector):

    setting = Typed(dict, ())

    def append_item(self, item):
        if item not in self.setting:
            self.setting[item.name] = item.default
        super(SingleSetting, self).append_item(item)

    def get_iterator(self, cycles=None):
        if cycles is None:
            return itertools.cycle([self.setting.copy()])
        else:
            return [self.setting.copy()]*cycle

    def get_value(self, item):
        return self.setting[item.name]

    def set_value(self, item, value):
        self.setting[item.name] = item.coerce_to_type(value)
        self.updated = True


class SequenceSelector(BaseSelector):

    settings = ContainerList(default=[])
    order = d_(Enum(*choice.options.keys()))

    def add_setting(self, values=None):
        if values is None:
            values = {}
        for item in self.context_items:
            if item not in values:
                values[item.name] = item.default
        self.settings.append(values)
        self.updated = True

    def remove_setting(self, setting):
        self.settings.remove(setting)
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
        value = item.coerce_to_type(value)
        self.settings[setting_index][item.name] = value
        self.updated = True

    def get_value(self, setting_index, item):
        return self.settings[setting_index][item.name]
