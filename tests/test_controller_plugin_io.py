from types import SimpleNamespace

import pytest

from psi.controller.api import NullOutput
from psi.controller.engines.null import NullEngine
from psi.controller.plugin import find_engines, find_outputs


def _fake_point(*extensions):
    return SimpleNamespace(extensions=list(extensions))


def _fake_extension(children):
    return SimpleNamespace(get_children=lambda cls: list(children))


def test_find_engines_duplicate_name_raises():
    # Regression: this error path used to raise NameError (undefined
    # `engine_error`) instead of the intended descriptive ValueError.
    e1 = NullEngine(name='dup', buffer_size=10)
    e2 = NullEngine(name='dup', buffer_size=10)
    point = _fake_point(_fake_extension([e1, e2]))
    with pytest.raises(ValueError, match='More than one engine named "dup"'):
        find_engines(point)


def test_find_engines_last_engine_is_default_master():
    e1 = NullEngine(name='a', buffer_size=10, weight=0)
    e2 = NullEngine(name='b', buffer_size=10, weight=1)
    point = _fake_point(_fake_extension([e1, e2]))
    engines, master = find_engines(point)
    assert list(engines) == ['a', 'b']
    assert master is e2


def test_find_outputs_duplicate_name_raises(ao_channel):
    # Regression: this error path used to raise NameError (undefined
    # `output_error`) instead of the intended descriptive ValueError.
    NullOutput(name='dup', parent=ao_channel)
    NullOutput(name='dup', parent=ao_channel)
    point = _fake_point()
    with pytest.raises(ValueError, match='More than one output named "dup"'):
        find_outputs({'speaker': ao_channel}, point)
