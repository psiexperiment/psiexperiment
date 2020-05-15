import pytest
import numpy as np

from psi.context.api import Parameter, EnumParameter


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
