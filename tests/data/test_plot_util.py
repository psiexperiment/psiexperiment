"""Tests for psi.data.plot_util (pure plotting math, no Qt)."""
import numpy as np

from psi.data.plot_util import (
    decimate_extremes, decimate_mean, format_log_ticks, format_time,
    get_color_cycle, get_freq,
)


def test_get_freq():
    freq = get_freq(1000, 1)
    assert freq[0] == 0
    assert freq[-1] == 500
    assert len(freq) == 501


def test_format_time():
    assert format_time(0) == '00:00:00'
    assert format_time(61) == '00:01:01'
    assert format_time(3661) == '01:01:01'
    assert format_time(90, fmt='{M:02d}:{S:02.0f}') == '01:30'


def test_format_log_ticks():
    # Values arrive as log10; output is in kHz with one decimal.
    assert format_log_ticks([3, 4], None, None) == ['1.0', '10.0']


def test_get_color_cycle():
    colors = list(get_color_cycle('palettable.colorbrewer.qualitative.Set1_9', 3))
    assert len(colors) == 3
    for color in colors:
        assert len(color) == 4  # RGBA
        assert all(0 <= v <= 255 for v in color)
    assert len(set(colors)) == 3  # distinct


def test_decimate_mean_1d():
    data = np.array([1.0, 3.0, 2.0, 4.0, 10.0])
    # Downsample of 2 discards the trailing fragment (10.0).
    result = decimate_mean(data, 2)
    np.testing.assert_array_equal(result, [2.0, 3.0])


def test_decimate_mean_2d():
    data = np.array([[1.0, 3.0, 2.0, 4.0],
                     [10.0, 30.0, 20.0, 40.0]])
    result = decimate_mean(data, 2)
    np.testing.assert_array_equal(result, [[2.0, 3.0], [20.0, 30.0]])


def test_decimate_mean_empty():
    # Regression: the empty case used to return a two-element tuple
    # (copy-paste from decimate_extremes), breaking len()/isnan() handling
    # in ChannelPlot.update.
    result = decimate_mean(np.array([]), 2)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_decimate_extremes_1d():
    data = np.array([1.0, 3.0, 2.0, 4.0, 10.0])
    d_min, d_max = decimate_extremes(data, 2)
    np.testing.assert_array_equal(d_min, [1.0, 2.0])
    np.testing.assert_array_equal(d_max, [3.0, 4.0])


def test_decimate_extremes_empty():
    d_min, d_max = decimate_extremes(np.array([]), 2)
    assert d_min.size == 0
    assert d_max.size == 0


def test_decimate_preserves_input():
    data = np.array([1.0, 3.0, 2.0, 4.0])
    original = data.copy()
    decimate_mean(data, 2)
    decimate_extremes(data, 2)
    np.testing.assert_array_equal(data, original)


def test_prepare_curve_raw():
    from psi.data.plot_util import prepare_decimated_curve
    t = np.arange(4) / 4
    data = np.array([1.0, 2.0, 3.0, 4.0])
    x, y, kw = prepare_decimated_curve(data, t, downsample=1, mode='extremes')
    np.testing.assert_array_equal(x, t)
    np.testing.assert_array_equal(y, data)
    assert kw == {}


def test_prepare_curve_extremes():
    from psi.data.plot_util import prepare_decimated_curve
    t = np.arange(4.0)
    data = np.array([1.0, 3.0, 4.0, 2.0])
    x, y, kw = prepare_decimated_curve(data, t, downsample=2, mode='extremes')
    # Each decimation bin produces a (min, max) vertical segment.
    np.testing.assert_array_equal(x, [0, 0, 2, 2])
    np.testing.assert_array_equal(y, [1, 3, 2, 4])
    assert kw == {'connect': 'pairs'}


def test_prepare_curve_mean():
    from psi.data.plot_util import prepare_decimated_curve
    t = np.arange(4.0)
    data = np.array([1.0, 3.0, 4.0, 2.0])
    x, y, kw = prepare_decimated_curve(data, t, downsample=2, mode='mean')
    np.testing.assert_array_equal(x, [0, 2])
    np.testing.assert_array_equal(y, [2, 3])
    assert kw == {}


def test_prepare_curve_all_nan_clears_plot():
    from psi.data.plot_util import prepare_decimated_curve
    t = np.arange(4.0)
    data = np.full(4, np.nan)
    for mode, ds in [('extremes', 2), ('mean', 2), ('none', 1)]:
        x, y, kw = prepare_decimated_curve(data, t, downsample=ds, mode=mode)
        assert x.size == 0 and y.size == 0


def test_prepare_curve_mismatched_shapes_returns_none():
    from psi.data.plot_util import prepare_decimated_curve
    # Time axis shorter than the decimated data: transient buffer-resize
    # state; the caller must skip the update.
    t = np.arange(1.0)
    data = np.arange(8.0)
    assert prepare_decimated_curve(data, t, downsample=2, mode='mean') is None
