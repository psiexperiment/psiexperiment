'''
Machinery for wiring stimulus token blocks into the context system.

A token (see :mod:`psi.token`) is a declarative tree of blocks (e.g., a tone
nested inside an envelope), where each block contributes parameters (e.g.,
frequency, rise time). When a token is assigned to an output, each block
parameter is registered with the context plugin under a globally unique name
(``{output}_{block}_{parameter}``). At stimulus-generation time, the values
for those context items are mapped back to each block's own parameter names
and used to instantiate the block's waveform factory.

The mapping between global context names and block parameter names is stored
on the output itself (see ``BaseOutput._block_context_map``), keyed by block.
Historically this was a module-level global keyed by ``(output, block)`` that
was never cleaned up.
'''
import logging
log = logging.getLogger(__name__)

import copy


def load_items(output, block):
    '''
    Create the context items for all parameters in the block hierarchy.

    Each parameter is copied and renamed to ``{output}_{block}_{parameter}``
    so it can be registered with the context plugin without colliding with
    other outputs using the same token. The mapping from the new (global)
    name back to the block's own parameter name is recorded on the output.

    Parameters
    ----------
    output : BaseOutput
        Output the token is assigned to.
    block : Block or None
        Root block of the token.

    Returns
    -------
    parameters : list of Parameter
        Renamed copies of all parameters in the block hierarchy, ready to be
        contributed to the context plugin.
    '''
    if block is None:
        return []
    # Rebuild the map from scratch so entries from a previously-assigned
    # token do not linger.
    output._block_context_map = {}
    return _load_block_items(output, block)


def _load_block_items(output, block):
    from .output import ContinuousOutput
    scope = 'experiment' if isinstance(output, ContinuousOutput) else 'trial'

    block_map = {}
    parameters = []
    for parameter in block.parameters:
        new_parameter = copy.copy(parameter)
        new_parameter.name = '{}_{}_{}' \
            .format(output.name, block.name, parameter.name)
        new_parameter.label = '{} {}' \
            .format(block.label, parameter.label)
        new_parameter.compact_label = '{} {}' \
            .format(block.compact_label, parameter.compact_label)
        new_parameter.group_name = output.name
        new_parameter.scope = scope
        parameters.append(new_parameter)
        block_map[new_parameter.name] = parameter.name

    output._block_context_map[block] = block_map

    for b in block.blocks:
        b_params = _load_block_items(output, b)
        parameters.extend(b_params)

    return parameters


def get_parameters(output, block):
    '''
    Return the (global) context item names for all parameters in the block
    hierarchy. Requires `load_items` to have been called first.
    '''
    if block is None:
        return []
    parameters = list(_get_block_map(output, block).keys())
    for b in block.blocks:
        parameters.extend(get_parameters(output, b))
    return parameters


def _get_block_map(output, block):
    try:
        return output._block_context_map[block]
    except KeyError as e:
        raise KeyError(
            f'Token parameters for output "{output.name}" have not been '
            f'registered. Ensure the output manifest is loaded (via '
            f'load_items) before requesting stimulus generation.'
        ) from e


def initialize_factory(output, block, context):
    '''
    Instantiate the waveform factory for the block hierarchy.

    Parameters
    ----------
    output : BaseOutput
        Output the token is assigned to.
    block : Block
        Root block of the token.
    context : dict
        Mapping of global context item names to values. May also provide
        `fs` and `calibration`; if absent, they are read from the output.
    '''
    input_factories = [initialize_factory(output, b, context)
                       for b in block.blocks]

    # Pull out list of params accepted by factory class so we can figure out
    # if there's anything important that needs to be added to the context
    # (e.g., sampling rate).
    code = block.factory.__init__.__code__
    params = code.co_varnames[1:code.co_argcount]

    context = context.copy()
    if 'fs' not in context:
        context['fs'] = output.fs
        context['calibration'] = output.calibration

    # Now, pull out the block-specific context.
    block_context = {bn: context[gn] for gn, bn in
                     _get_block_map(output, block).items()}

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

    # Values set by `values` attribute on the block override anything we can
    # pull in.
    block_context.update(block.values)
    return block.factory(**block_context)


def generate_waveform(output, context):
    factory = initialize_factory(output, output.token, context)
    return factory.get_samples_remaining()
