import itertools
import operator
import collections
from copy import deepcopy

from atom.api import ContainerList, Typed, Enum, Event
from enaml.core.declarative import Declarative

from . import choice
from .. import SimpleState


class BaseSelector(SimpleState, Declarative):

    parameters = ContainerList()
    updated = Event()

    def append_parameter(self, parameter):
        self.insert_parameter(len(self.parameters), parameter)
        self.updated = True

    def remove_parameter(self, parameter):
        self.parameters.remove(parameter)
        self.updated = True

    def insert_parameter(self, index, parameter):
        self.parameters.insert(index, parameter)
        self.updated = True

    def find_parameter(self, name):
        for p in self.parameters:
            if p.name == name:
                return p

    def move_parameter(self, parameter, after=None):
        if parameter == after:
            return
        self.parameters.remove(parameter)
        if after is None:
            index = 0
        else:
            index = self.parameters.index(after)+1
        self.parameters.insert(index, parameter)
        self.updated = True


class SingleSetting(BaseSelector):

    setting = Typed(dict, ())

    def insert_parameter(self, index, parameter):
        if parameter.name not in self.setting:
            self.setting[parameter.name] = parameter.default
        super(SingleSetting, self).insert_parameter(index, parameter)

    def get_iterator(self):
        return itertools.cycle([self.setting.copy()])

    def get_value(self, parameter_name):
        return self.setting[parameter_name]

    def set_value(self, parameter_name, value):
        for p in self.parameters:
            if p.name == parameter_name:
                value = p.dtype(value)
                break
        self.setting[parameter_name] = value
        #self.setting = self.setting.copy()
        self.updated = True

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        attrs = ['parameters', 'setting']
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True


class SequenceSelector(BaseSelector):

    settings = ContainerList()
    order = Enum(*choice.options.values())

    def _default_settings(self):
        return []

    def add_setting(self, values=None):
        if values is None:
            values = {}
        for p in self.parameters:
            if p.name not in values:
                values[p.name] = p.default
        self.settings.append(values)
        self.updated = True

    def remove_setting(self, setting):
        self.settings.remove(setting)
        self.updated = True

    def insert_parameter(self, index, parameter):
        for setting in self.settings:
            if parameter.name not in setting:
                setting[parameter.name] = parameter.default
        super(SequenceSelector, self).insert_parameter(index, parameter)

    def sort_settings(self):
        names = [p.name for p in self.parameters]
        self.settings.sort(key=operator.itemgetter(*names))
        self.updated = True

    def get_iterator(self):
        # Some selectors need to sort the settings. To make sure that the
        # selector sorts the parameters in the order the columns are specified,
        # we need to use an OrderedDict.
        names = [p.name for p in self.parameters]
        ordered_settings = []
        for setting in self.settings:
            ordered_setting = collections.OrderedDict()
            for n in names:
                ordered_setting[n] = setting[n]
            ordered_settings.append(ordered_setting)
        return self.order(ordered_settings)

    def set_value(self, setting_index, parameter_name, value):
        for p in self.parameters:
            if p.name == parameter_name:
                value = p.dtype(value)
                break
        self.settings[setting_index][parameter_name] = value
        self.updated = True

    def get_value(self, setting_index, parameter_name):
        return self.settings[setting_index][parameter_name]

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        attrs = ['parameters', 'settings', 'order']
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True
