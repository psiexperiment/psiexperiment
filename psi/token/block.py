import copy

from atom.api import Callable, Dict, List, Property, Typed, Str

from enaml.core.api import Declarative, d_

from psi.context.api import Parameter


class Block(Declarative):

    name = d_(Str())
    label = d_(Str())
    compact_label = d_(Str())
    factory = d_(Callable())
    context_name_map = Typed(dict)

    blocks = Property()
    parameters = Property()

    hide = d_(List())
    values = d_(Dict())

    def initialize(self):
        super().initialize()
        names = []
        for p in self.parameters:
            names.append(p.name)
            if p.name in self.hide:
                p.visible = False
            if p.name in self.values:
                p.visible = False
        for h in self.hide:
            if h not in names:
                raise AttributeError('Cannot hide %s. Parameter not in %s.', h, self.name)
        for v in self.values.keys():
            if v not in names:
                raise AttributeError('Cannot set value for %s. Parameter not in %s.', h, self.name)

    def get_children(self, child_type):
        return [c for c in self.children if isinstance(c, child_type)]

    def _get_blocks(self):
        return self.get_children(Block)

    def _get_parameters(self):
        return self.get_children(Parameter)


class EpochBlock(Block):
    pass


class ContinuousBlock(Block):
    pass
