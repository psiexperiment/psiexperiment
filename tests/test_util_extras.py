import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from atom.api import Atom, Float, Int, Str, Value

from psi.util import (
    PSIJsonEncoder, copy_declarative, declarative_to_dict,
    declarative_to_json, dict_to_declarative, get_dependencies,
    get_tagged_members, get_tagged_values, psi_json_decoder_hook,
)


# -------- get_dependencies --------

def test_get_dependencies_simple_names():
    deps = get_dependencies('a + b')
    assert set(deps) == {'a', 'b'}


def test_get_dependencies_single_name():
    assert get_dependencies('event_name') == ['event_name']


def test_get_dependencies_attribute_access():
    deps = get_dependencies('np.random.randint(x)')
    assert 'np.random.randint' in deps
    assert 'x' in deps


def test_get_dependencies_call():
    deps = get_dependencies('foo(bar, baz=qux)')
    assert set(deps) >= {'foo', 'bar', 'qux'}


# -------- tagged member helpers --------

class _Tagged(Atom):
    a = Value(1)
    b = Value(2).tag(preference=True)
    c = Value(3).tag(preference=True, metadata=True)
    d = Value(4).tag(metadata=True)


def test_get_tagged_members_returns_matching():
    obj = _Tagged()
    assert set(get_tagged_members(obj, 'preference')) == {'b', 'c'}
    assert set(get_tagged_members(obj, 'metadata')) == {'c', 'd'}


def test_get_tagged_values_returns_current_values():
    obj = _Tagged()
    obj.b = 22
    obj.d = 44
    assert get_tagged_values(obj, 'preference') == {'b': 22, 'c': 3}
    assert get_tagged_values(obj, 'metadata') == {'c': 3, 'd': 44}


# -------- declarative_to_dict / dict_to_declarative --------

class _MetaAtom(Atom):
    name = Str().tag(metadata=True)
    value = Float(1.5).tag(metadata=True)
    untagged = Int(7)


def test_declarative_to_dict_primitives():
    assert declarative_to_dict(1, 'metadata') == 1
    assert declarative_to_dict(1.5, 'metadata') == 1.5
    assert declarative_to_dict('hi', 'metadata') == 'hi'
    assert declarative_to_dict(None, 'metadata') is None


def test_declarative_to_dict_path():
    assert declarative_to_dict(Path('/tmp/foo'), 'metadata') == str(Path('/tmp/foo'))


def test_declarative_to_dict_ndarray():
    assert declarative_to_dict(np.array([1, 2, 3]), 'metadata') == [1, 2, 3]


def test_declarative_to_dict_atom_with_dunder():
    obj = _MetaAtom(name='hello', value=2.5)
    result = declarative_to_dict(obj, 'metadata')
    assert result['__type__'] == '_MetaAtom'
    assert result['name'] == 'hello'
    assert result['value'] == 2.5
    # Untagged members are excluded.
    assert 'untagged' not in result


def test_declarative_to_dict_atom_without_dunder():
    obj = _MetaAtom(name='hello')
    result = declarative_to_dict(obj, 'metadata', include_dunder=False)
    assert '__type__' not in result
    assert '__id__' not in result
    assert result['name'] == 'hello'


def test_declarative_to_dict_list_and_dict():
    payload = [_MetaAtom(name='a'), {'inner': _MetaAtom(name='b')}]
    out = declarative_to_dict(payload, 'metadata', include_dunder=False)
    assert out[0] == {'name': 'a', 'value': 1.5}
    assert out[1]['inner'] == {'name': 'b', 'value': 1.5}


def test_declarative_to_dict_handles_cycles():
    # Two Atom instances referencing each other via list/dict containers.
    obj = _MetaAtom(name='root')
    seen = []
    # First call records the id; second call collapses to the __obj__:: marker.
    first = declarative_to_dict(obj, 'metadata', seen_objects=seen)
    second = declarative_to_dict(obj, 'metadata', seen_objects=seen)
    assert isinstance(first, dict)
    assert second == f'__obj__::{id(obj)}'


def test_dict_to_declarative_roundtrip():
    src = _MetaAtom(name='x', value=9.0)
    payload = declarative_to_dict(src, 'metadata', include_dunder=False)
    dst = _MetaAtom()
    dict_to_declarative(dst, payload)
    assert dst.name == 'x'
    assert dst.value == 9.0


def test_declarative_to_json_writes_file(tmp_path):
    obj = _MetaAtom(name='persist', value=3.14)
    out = tmp_path / 'meta.json'
    declarative_to_json(out, obj, 'metadata', include_dunder=False)
    parsed = json.loads(out.read_text())
    assert parsed == {'name': 'persist', 'value': 3.14}


# -------- copy_declarative --------

def test_copy_declarative_copies_metadata_members():
    src = _MetaAtom(name='orig', value=2.0)
    src.untagged = 99
    copy = copy_declarative(src)
    assert copy.name == 'orig'
    assert copy.value == 2.0
    # Untagged members get their class default, not the runtime value.
    assert copy.untagged == 7


def test_copy_declarative_excludes_and_overrides():
    src = _MetaAtom(name='orig', value=2.0)
    copy = copy_declarative(src, exclude=['value'], name='renamed')
    assert copy.name == 'renamed'
    assert copy.value == 1.5  # back to default since excluded


# -------- json encoder / decoder --------

def test_json_encoder_numpy_scalars():
    s = json.dumps(
        {
            'i': np.int64(7),
            'f': np.float32(1.5),
            'b': np.bool_(True),
            'arr': np.array([1, 2, 3]),
        },
        cls=PSIJsonEncoder,
    )
    parsed = json.loads(s)
    assert parsed == {'i': 7, 'f': 1.5, 'b': 1, 'arr': [1, 2, 3]}


def test_json_encoder_path_uuid_slice():
    u = uuid.uuid4()
    s = json.dumps(
        {'p': Path('/tmp/x'), 'u': u, 's': slice(0, 5, 2)},
        cls=PSIJsonEncoder,
    )
    parsed = json.loads(s)
    assert parsed['p'] == str(Path('/tmp/x'))
    assert parsed['u'] == str(u)
    assert parsed['s'] == 'slice(0, 5, 2)'


def test_json_decoder_hook_legacy_ndarray():
    payload = {'__ndarray__': [1, 2, 3], 'dtype': 'int64'}
    result = psi_json_decoder_hook(payload)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype('int64')
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_json_decoder_hook_passthrough():
    assert psi_json_decoder_hook({'foo': 1}) == {'foo': 1}
