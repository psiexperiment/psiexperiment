import pytest

from atom.api import Atom, Value

from psi.util import get_tagged_values, wrap_text


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



text1 = '''
Lorem ipsum dolor

Sit amet consectetur adipiscing elit.

Quisque faucibus ex sapien vitae pellentesque sem placerat.
In id cursus mi
pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas.
'''

text2 = '''
    Lorem ipsum dolor

    Sit amet consectetur adipiscing elit.

    Quisque faucibus ex sapien vitae pellentesque sem placerat.
    In id cursus mi
    pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas.
    '''

text3 = '''
    Lorem ipsum dolor

    Sit amet consectetur adipiscing elit.

    Quisque faucibus ex sapien vitae pellentesque sem placerat.
    In id cursus mi
    pretium tellus duis convallis. Tempus leo eu aenean sed diam urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum egestas.
    '''

expected_text = '''Lorem ipsum dolor

Sit amet consectetur adipiscing elit.

Quisque faucibus ex sapien vitae pellentesque sem placerat. In id
cursus mi pretium tellus duis convallis. Tempus leo eu aenean sed diam
urna tempor. Pulvinar vivamus fringilla lacus nec metus bibendum
egestas.'''

@pytest.mark.parametrize('text', [text1, text2, text3])
def test_wrap_text(text):
    assert wrap_text(text) == expected_text
