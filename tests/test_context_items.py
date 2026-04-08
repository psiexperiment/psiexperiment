import pytest
import numpy as np

from psi.context.api import Parameter, EnumParameter, MultiSelectParameter


def test_enum_parameter():
    item = EnumParameter(
        name='polarity',
        choices = {'positive': 1, 'negative': -1}
    )
    assert item.label == item.name
    assert item.compact_label == item.name
    assert item.dtype == np.dtype('int64').str
    item.selected = 'negative'
    assert item.expression == -1


def test_parameter():
    item = Parameter(
        name='frequency',
        default=100.0
    )
    assert item.label == item.name
    assert item.compact_label == item.name
    assert item.dtype == np.dtype('float').str
    assert item.coerce_to_type('100') == 100.0
    assert item.coerce_to_type(100) == 100.0


def test_multi_select():
    item = MultiSelectParameter(
        name='stimuli',
        choices={
            'NBN (2 kHz)': 'nbn_2',
            'NBN (4 kHz)': 'nbn_4',
            'Silence': 'silence',
            'SAM (2 kHz)': 'sam_2',
            'SAM (4 kHz)': 'sam_4',
        }
    )

    assert item.selected == []
    assert item.expression == '[]'
    item.expression = "['nbn_2', 'nbn_4']"
    assert item.selected == ['NBN (2 kHz)', 'NBN (4 kHz)']

    item.selected = ['Silence', 'SAM (2 kHz)']
    assert item.expression == "['sam_2', 'silence']"

    # Verify ordering does not matter for expression
    item.selected = ['SAM (2 kHz)', 'Silence']
    assert item.expression == "['sam_2', 'silence']"
