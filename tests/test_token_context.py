"""Unit tests for psi.controller.token_context (extracted from
output_manifest.enaml). These run without a workbench."""
import pytest

from psi.controller.api import EpochOutput
from psi.controller.token_context import (
    get_parameters, initialize_factory, load_items,
)
from psi.token.api import Cos2Envelope, Tone


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
