"""Unit tests for psi.controller.token_context (extracted from
output_manifest.enaml). These run without a workbench."""
import enaml
import pytest

from psi.controller.api import EpochOutput
from psi.controller.token_context import (
    get_parameters, initialize_factory, load_items,
)
from psi.context.api import EnumParameter
from psi.token.api import Cos2Envelope, Tone

with enaml.imports():
    from psi.token.primitives import BandlimitedFIRNoise


@pytest.fixture
def tone_burst():
    token = Cos2Envelope(name='tone_burst')
    Tone(parent=token)
    return token


@pytest.fixture
def output():
    return EpochOutput(name='target')


def test_load_items_renames_parameters(output, tone_burst):
    parameters = load_items(output, tone_burst)
    names = {p.name for p in parameters}
    assert names == {
        'target_tone_burst_start_time',
        'target_tone_burst_duration',
        'target_tone_burst_rise_time',
        'target_tone_level',
        'target_tone_frequency',
        'target_tone_polarity',
        'target_tone_phase',
    }
    # Epoch outputs get trial scope.
    assert all(p.scope == 'trial' for p in parameters)


def test_load_items_none_block(output):
    assert load_items(output, None) == []


def test_get_parameters_requires_load_items(output, tone_burst):
    with pytest.raises(KeyError, match='have not been registered'):
        get_parameters(output, tone_burst)


def test_initialize_factory_requires_load_items(output, tone_burst):
    with pytest.raises(KeyError, match='have not been registered'):
        initialize_factory(output, tone_burst, {'fs': 1000})


def test_get_parameters_after_load(output, tone_burst):
    load_items(output, tone_burst)
    names = set(get_parameters(output, tone_burst))
    assert 'target_tone_frequency' in names
    assert 'target_tone_burst_rise_time' in names


def test_load_items_rebuilds_map_on_token_swap(output, tone_burst):
    # Assigning a new token must not leave stale entries from the old one.
    load_items(output, tone_burst)
    old_blocks = set(output._block_context_map)

    replacement = Cos2Envelope(name='tone_burst')
    Tone(parent=replacement)
    load_items(output, replacement)

    new_blocks = set(output._block_context_map)
    assert old_blocks.isdisjoint(new_blocks)
    assert len(new_blocks) == len(old_blocks)


def test_maps_are_per_output(tone_burst):
    # Two outputs sharing a token structure must not interfere.
    a = EpochOutput(name='a')
    b = EpochOutput(name='b')
    load_items(a, tone_burst)
    assert a._block_context_map
    assert not b._block_context_map


@pytest.fixture
def fixed_noise():
    '''
    A noise burst whose paradigm fixes some of its settings (`values`), of
    each kind: plain parameters and drop-downs whose choices are Python
    values or expressions.
    '''
    token = Cos2Envelope(name='noise_burst', values={'start_time': 0.5})
    BandlimitedFIRNoise(parent=token, values={
        'polarity': -1,
        'legacy_level': True,
        'audiogram_weighting': 'mouse',
    })
    return token


def items_by_name(output, token):
    return {p.name: p for p in load_items(output, token)}


@pytest.mark.parametrize('name, expression', [
    ('target_noise_burst_start_time', '0.5'),
    ('target_bandlimited_noise_polarity', '-1'),
    # Drop-downs hold the choice's value as their expression.
    ('target_bandlimited_noise_legacy_level', True),
    ('target_bandlimited_noise_audiogram_weighting', "'mouse'"),
])
def test_load_items_fixes_values(output, fixed_noise, name, expression):
    '''
    The context item of a fixed setting holds the fixed value (which is what
    the stimulus is built with and what gets recorded), not its default, and
    can't be changed by the user or by saved preferences.
    '''
    item = items_by_name(output, fixed_noise)[name]
    assert item.expression == expression
    assert not item.editable
    assert not item.visible


def test_load_items_leaves_other_values(output, fixed_noise):
    items = items_by_name(output, fixed_noise)
    item = items['target_bandlimited_noise_equalize']
    assert item.selected == 'yes'
    assert item.editable


def test_load_items_does_not_change_token(output, fixed_noise):
    '''
    The token's own parameters keep their defaults: the context items are
    copies, and other outputs may use the same token.
    '''
    load_items(output, fixed_noise)
    noise, = fixed_noise.blocks
    legacy_level, = [p for p in noise.parameters
                     if p.name == 'legacy_level']
    assert legacy_level.selected == 'no'
    assert legacy_level.editable


def test_load_items_rejects_fixed_value_not_a_choice(output):
    token = Cos2Envelope(name='noise_burst')
    BandlimitedFIRNoise(parent=token, values={'audiogram_weighting': 'rat'})
    with pytest.raises(ValueError, match='Cannot fix .*audiogram_weighting'):
        load_items(output, token)


def test_enum_fix_value_tells_true_from_one():
    '''
    True == 1 in Python, so a drop-down offering both must not take one for
    the other.
    '''
    item = EnumParameter(name='x', choices={'one': 1, 'true': True},
                         default='one')
    item.fix_value(True)
    assert item.selected == 'true'
    item.fix_value(1)
    assert item.selected == 'one'
