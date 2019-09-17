import pytest

from atom.api import Atom, Value

from psi.util import get_tagged_values


class PreferencesContainer(Atom):

    a = Value(1)
    b = Value(2).tag(preference=True)
    c = Value(3).tag(not_preference=True)
    d = Value(4).tag(preference=True)


@pytest.fixture
def preferences():
    return PreferencesContainer()


def test_get_tagged_values(preferences):
    result = get_tagged_values(preferences, 'preference')
    assert result == {'b': 2, 'd': 4}
