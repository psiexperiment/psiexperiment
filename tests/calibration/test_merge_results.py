"""Tests for psi.controller.calibration.calibrate.merge_results."""
import pandas as pd
import pytest

from psi.controller.calibration.calibrate import merge_results


def _df(spl, freqs):
    df = pd.DataFrame({'spl': spl, 'freq': freqs}).set_index('freq')
    return df


def test_merge_dataframes_concats_with_outer_level():
    a = _df([60, 70, 80], [1000, 2000, 4000])
    b = _df([55, 65, 75], [1000, 2000, 4000])
    merged = merge_results({('ao_a',): a, ('ao_b',): b})
    cal = merged['calibration']
    # The new outer level is named after the `names` argument default.
    assert 'ao_channel' in cal.index.names
    assert sorted(cal.index.get_level_values('ao_channel').unique()) == ['ao_a', 'ao_b']
    # The data itself round-trips.
    assert cal.loc[('ao_a', 1000), 'spl'] == 60
    assert cal.loc[('ao_b', 4000), 'spl'] == 75


def test_merge_strips_per_frame_attrs_before_concat():
    # pandas refuses to concat DataFrames with conflicting `.attrs` unless they
    # are cleared. merge_results clears them explicitly; verify the call
    # succeeds when attrs differ.
    a = _df([60], [1000])
    a.attrs['vrms'] = {'value': 1.0}
    b = _df([55], [1000])
    b.attrs['vrms'] = {'value': 2.0}
    merged = merge_results({('ao_a',): a, ('ao_b',): b})
    assert 'calibration' in merged


def test_merge_rejects_unmergeable_attr():
    a = _df([60], [1000])
    a.attrs['weird'] = 42  # scalar — neither DataFrame nor dict
    with pytest.raises(ValueError, match='Unable to merge'):
        merge_results({('ao_a',): a})


def test_merge_respects_custom_names():
    a = _df([60], [1000])
    b = _df([55], [1000])
    merged = merge_results({('a',): a, ('b',): b}, names=['speaker'])
    assert 'speaker' in merged['calibration'].index.names
