import numpy as np

from psi import get_config


def test_get_config_default_returned_for_missing_setting():
    default = 'fallback'
    assert get_config('__no_such_setting__', default) == default


def test_get_config_none_is_valid_default():
    assert get_config('__no_such_setting__', None) is None


def test_get_config_array_default_identity():
    # Regression: the sentinel check used `!=` instead of `is not`, so any
    # default whose __eq__ returns a non-scalar (e.g. numpy arrays) raised
    # "truth value is ambiguous" inside get_config itself.
    default = np.array([1.0, 2.0])
    result = get_config('__no_such_setting__', default)
    assert result is default
