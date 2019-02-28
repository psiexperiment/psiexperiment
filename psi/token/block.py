import copy

from atom.api import Callable, List, Property, Typed, Unicode

from enaml.core.api import Declarative, d_, d_func

from psi.context.api import Parameter


class Block(Declarative):

    name = d_(Unicode())
    label = d_(Unicode())
    compact_label = d_(Unicode())
    factory = d_(Callable())
    context_name_map = Typed(dict)

    blocks = Property()
    parameters = Property()

    hide = d_(List())

    def initialize(self):
        super().initialize()
        for p in self.parameters:
            if p.name in self.hide:
                p.visible = False

    def get_children(self, child_type):
        return [c for c in self.children if isinstance(c, child_type)]

    def _get_blocks(self):
        return self.get_children(Block)

    def _get_parameters(self):
        return self.get_children(Parameter)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.name = self.name
        result.label = self.label
        result.compact_label = self.compact_label
        result.factory = self.factory
        for c in self.children:
            result.children.append(copy.copy(c))
        return result


class EpochBlock(Block):
    pass


class ContinuousBlock(Block):
    pass
