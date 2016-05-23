import itertools
import operator
import collections
from copy import deepcopy

from atom.api import ContainerList, Typed, Enum, Event
from enaml.core.declarative import Declarative

from . import choice
from .plugin import ContextPlugin
from .. import SimpleState


class BaseSelector(SimpleState, Declarative):

    context_plugin = Typed(ContextPlugin).tag(transient=True)
    context_items = Typed(list, [])
    updated = Event()

    def append_item(self, item_name):
        context_items = self.context_items[:]
        context_items.append(item_name)
        self.context_items = context_items
        self.updated = True

    def remove_item(self, item_name):
        context_items = self.context_items[:]
        context_items.remove(item_name)
        self.context_items = context_items
        self.updated = True

    def get_item_info(self, item_name, attribute):
        return self.context_plugin.get_item_info(item_name)[attribute]


class SingleSetting(BaseSelector):

    setting = Typed(dict, ())

    def append_item(self, item_name):
        if item_name not in self.setting:
            self.setting[item_name] = self.get_item_info(item_name, 'default')
        super(SingleSetting, self).append_item(item_name)

    def get_iterator(self):
        return itertools.cycle([self.setting.copy()])

    def get_value(self, item_name):
        return self.setting[item_name]

    def set_value(self, item_name, value):
        dtype = self.get_item_info(item_name, 'dtype')
        self.setting[item_name] = dtype(value)
        self.updated = True

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        attrs = ['context_items', 'setting']
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


class SequenceSelector(BaseSelector):

    settings = ContainerList(default=[])
    order = Enum(*choice.options.values())

    def add_setting(self, values=None):
        if values is None:
            values = {}
        for item_name in self.context_items:
            if item_name not in values:
                default = self.get_item_info(item_name, 'default')
                values[item_name] = default
        self.settings.append(values)
        self.updated = True

    def remove_setting(self, setting):
        self.settings.remove(setting)
        self.updated = True

    def append_item(self, item_name):
        for setting in self.settings:
            if item_name not in setting:
                default = self.get_item_info(item_name, 'default')
                setting[item_name] = default
        super(SequenceSelector, self).append_item(item_name)

    def sort_settings(self):
        self.settings.sort()
        self.updated = True

    def get_iterator(self):
        # Some selectors need to sort the settings. To make sure that the
        # selector sorts the parameters in the order the columns are specified,
        # we need to use an OrderedDict.
        ordered_settings = []
        for setting in self.settings:
            ordered_setting = collections.OrderedDict()
            for item_name in self.context_items:
                ordered_setting[item_name] = setting[item_name]
            ordered_settings.append(ordered_setting)
        return self.order(ordered_settings)

    def set_value(self, setting_index, item_name, value):
        dtype = self.get_item_info(item_name, 'dtype')
        self.settings[setting_index][item_name] = dtype(value)
        self.updated = True

    def get_value(self, setting_index, item_name):
        return self.settings[setting_index][item_name]

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        attrs = ['context_items', 'settings', 'order']
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True
