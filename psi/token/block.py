import copy

from atom.api import Unicode, Callable, Typed, Property

from enaml.core.api import Declarative, d_, d_func

from psi.context.api import Parameter


class Block(Declarative):

    name = d_(Unicode())
    label = d_(Unicode())
    factory = d_(Callable())
    context_name_map = Typed(dict)

    blocks = Property()
    parameters = Property()

    _block_context = Typed(dict)

    def get_children(self, child_type):
        return [c for c in self.children if isinstance(c, child_type)]

    def _get_blocks(self):
        return self.get_children(Block)

    def _get_parameters(self):
        return self.get_children(Parameter)

    def configure_context_items(self, output_name, output_label, scope):
        self.context_name_map = {}
        for item in self.parameters:
            old_name = item.name
            item.name = '{}_{}_{}'.format(output_name, self.name, item.name)
            item.label = '{} {}'.format(self.label, item.label)
            item.compact_label = '{} {} {}'.format(output_label, self.label,
                                                   item.compact_label)
            item.group = output_name
            item.scope = scope
            self.context_name_map[item.name] = old_name

        for block in self.blocks:
            block.configure_context_items(output_name, output_label, scope)

    def get_context_items(self):
        items = self.get_children(Parameter)
        for block in self.get_children(Block):
            items.extend(block.get_context_items())
        return items

    def get_context_names(self):
        return [item.name for item in self.get_context_items()]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.name = self.name
        result.label = self.label
        result.factory = self.factory
        for c in self.children:
            result.children.append(copy.copy(c))
        return result

    def initialize_factory(self, context):
        input_factories = [b.initialize_factory(context) for b in self.blocks]

        # Pull out list of params accepted by factory class
        code = self.factory.__init__.__code__
        params = code.co_varnames[1:code.co_argcount]
        block_context = self.get_block_context(context)

        # TODO: Is this a hack?
        if 'fs' in params:
            block_context['fs'] = context['fs']
        if 'calibration' in params:
            block_context['calibration'] = context['calibration']
        if 'input_factory' in params:
            if len(input_factories) != 1:
                raise ValueError('Incorrect number of inputs')
            block_context['input_factory'] = input_factories[0]
        if 'input_factories' in params:
            block_context['input_factories'] = input_factories
        return self.factory(**block_context)

    def get_block_context(self, context):
        return {bn: context[gn] for gn, bn in self.context_name_map.items()}


class EpochBlock(Block):
    pass


class ContinuousBlock(Block):
    pass
