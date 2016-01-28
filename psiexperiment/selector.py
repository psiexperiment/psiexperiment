import itertools
import operator
import collections

from atom.api import Atom, ContainerList, Dict, Enum, Typed, observe, Event

import choice


class BaseSelector(Atom):

    parameters = ContainerList()
    updated = Event()

    def _default_parameters(self):
        return []

    def append_parameter(self, parameter):
        self.insert_parameter(len(self.parameters), parameter)

    def remove_parameter(self, parameter):
        self.parameters.remove(parameter)

    def insert_parameter(self, index, parameter):
        self.parameters.insert(index, parameter)

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


class SingleSetting(BaseSelector):

    setting = Dict()

    def insert_parameter(self, index, parameter):
        if parameter.name not in self.setting:
            self.setting[parameter.name] = parameter.default_value
        super(SingleSetting, self).insert_parameter(index, parameter)

    def get_iterator(self):
        return itertools.cycle([self.setting.copy()])


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
                values[p.name] = p.default_value
        self.settings.append(values)

    def remove_setting(self, setting):
        self.settings.remove(setting)

    def insert_parameter(self, index, parameter):
        for setting in self.settings:
            if parameter.name not in setting:
                setting[parameter.name] = parameter.default_value
        super(SequenceSelector, self).insert_parameter(index, parameter)

    def sort_settings(self):
        names = [p.name for p in self.parameters]
        self.settings.sort(key=operator.itemgetter(*names))

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
