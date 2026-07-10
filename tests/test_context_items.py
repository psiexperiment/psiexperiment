import pytest
import numpy as np

from psi.context.api import (
    BoolParameter, EnumParameter, FileParameter, MultiSelectParameter,
    OrderedContextMeta, Parameter, UnorderedContextMeta,
)


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
    # Setter uses eval(), so input must be valid Python (quoted strings).
    # Getter respects quote_values (default False), so it returns unquoted.
    item.expression = "['nbn_2', 'nbn_4']"
    assert item.selected == ['NBN (2 kHz)', 'NBN (4 kHz)']

    item.selected = ['Silence', 'SAM (2 kHz)']
    assert item.expression == "[sam_2, silence]"

    # Verify ordering does not matter for expression
    item.selected = ['SAM (2 kHz)', 'Silence']
    assert item.expression == "[sam_2, silence]"


def test_parameter_preferences_roundtrip():
    src = Parameter(name='freq', default=100.0)
    src.expression = '200.0'
    src.rove = True
    prefs = src.get_preferences()
    assert prefs == {'expression': '200.0', 'rove': True}

    dst = Parameter(name='freq', default=100.0)
    dst.set_preferences(prefs)
    assert dst.expression == '200.0'
    assert dst.rove is True


def test_parameter_set_value_updates_expression():
    p = Parameter(name='gain', default=0.0)
    p.set_value(12.5)
    assert p.expression == '12.5'


def test_parameter_get_flags_reports_scope():
    p = Parameter(name='gain', default=0.0, scope='experiment')
    assert 'scope experiment' in p.get_flags()


def test_bool_parameter_dtype():
    p = BoolParameter(name='enabled', default=True)
    assert p.dtype == bool


def test_file_parameter_expression_roundtrip():
    p = FileParameter(name='source', path='/tmp/data.bin')
    assert p.expression == '"/tmp/data.bin"'
    p.expression = '"/other/path"'
    assert p.path == '/other/path'


def test_enum_parameter_invalid_expression_raises():
    # NOTE: the error-formatting code in EnumParameter._set_expression does
    # `', '.join(self.choices.values())`, so it only works when choice values
    # are strings. With int values, an unknown expression hits a TypeError
    # before the intended ValueError can be raised — see psi/context/context_item.py.
    item = EnumParameter(
        name='mode',
        choices={'fast': 'F', 'slow': 'S'},
    )
    with pytest.raises(ValueError, match='Could not map expression'):
        item.expression = 'NOT_A_CHOICE'


def test_enum_parameter_default_selected_falls_back_to_first():
    item = EnumParameter(
        name='mode',
        choices={'fast': 1, 'slow': 2},
        default='nonexistent',
    )
    # Falls back to the first key.
    assert item.selected == 'fast'


def test_multi_select_invalid_key_raises():
    item = MultiSelectParameter(
        name='stim',
        choices={'A': 'a', 'B': 'b'},
    )
    item.selected = ['A', 'NOPE']
    with pytest.raises(KeyError, match='Invalid key'):
        _ = item.expression


# -------- UnorderedContextMeta / OrderedContextMeta --------

def test_unordered_meta_add_remove_item():
    meta = UnorderedContextMeta(name='roving')
    meta.add_item('freq')
    meta.add_item('level')
    assert meta.values == {'freq', 'level'}
    meta.add_item('freq')  # idempotent
    assert meta.values == {'freq', 'level'}
    meta.remove_item('freq')
    assert meta.values == {'level'}
    meta.remove_item('not_there')  # silent no-op
    assert meta.values == {'level'}


def test_ordered_meta_respects_mandatory_and_forbidden():
    meta = OrderedContextMeta(
        name='ordered',
        mandatory_items=['freq'],
        forbidden_items=['hidden'],
    )
    # Mandatory items are present by default.
    assert 'freq' in meta.values

    # Forbidden items cannot be added.
    meta.set_choice('1', 'hidden')
    assert 'hidden' not in meta.values

    # Cannot drop a mandatory item.
    meta.set_choice(None, 'freq')
    assert 'freq' in meta.values


def test_ordered_meta_set_and_get_choice():
    meta = OrderedContextMeta(name='ordered')
    meta.set_choice('1', 'a')
    meta.set_choice('2', 'b')
    assert meta.get_choice('a') == '1'
    assert meta.get_choice('b') == '2'
    # Re-insert at position 1 moves 'b' to slot 1.
    meta.set_choice('1', 'b')
    assert meta.get_choice('b') == '1'
    assert meta.get_choice('a') == '2'


def test_ordered_meta_get_choices_lists():
    meta = OrderedContextMeta(name='ordered', mandatory_items=['m'])
    # m is mandatory, so no `None` option is offered.
    choices = meta.get_choices('m')
    assert None not in choices
    # An optional, not-yet-added item gets None plus the available positions.
    choices = meta.get_choices('q')
    assert None in choices
    # A forbidden item gets no choices at all.
    meta = OrderedContextMeta(name='ordered', forbidden_items=['x'])
    assert meta.get_choices('x') == []
