from types import SimpleNamespace

import numpy as np
import pytest

from psiaudio import util
from psiaudio.calibration import FlatCalibration

from psi.data.plot_groups import EpochGroupAccumulator
from psi.data.plots import (
    GroupedEpochAveragePlot, GroupedEpochFFTPlot, make_color,
)


def test_make_color():
    make_color('red')
    make_color('seagreen')
    make_color((0, 0, 0, 0))
    make_color((0, 0, 0))


class _FakeEpoch:

    def __init__(self, data):
        self._data = np.asarray(data, dtype=np.double)
        self.metadata = {'level': 60}
        self.shape = self._data.shape

    def __array__(self, dtype=None, copy=None):
        return self._data if dtype is None else self._data.astype(dtype)


def test_legacy_y_override_rejected():
    # Grouped epoch plots no longer call _y (they render from the running
    # mean); a stale override must fail loudly instead of being silently
    # ignored.
    class LegacyPlot(GroupedEpochAveragePlot):
        def _y(self, epoch):
            return epoch

    plot = LegacyPlot()
    with pytest.raises(TypeError, match='_render_mean'):
        plot.source = SimpleNamespace()


@pytest.mark.parametrize('average_mode', ['FFT', 'time'])
def test_fft_plot_incremental_matches_batch(average_mode):
    """The incremental fold/render path must reproduce the historical batch
    formula: psd over the epoch stack, calibration, mean over epochs."""

    class _Plot(GroupedEpochFFTPlot):
        # Bypass source wiring (callbacks, x-cache); only the math is
        # under test here.
        def _observe_source(self, event):
            pass

    fs = 1000
    n = 100
    calibration = FlatCalibration.from_db(20)
    rng = np.random.RandomState(3)
    epochs = rng.normal(size=(8, 1, n))
    freq = np.fft.rfftfreq(n, 1 / fs)

    plot = _Plot(average_mode=average_mode, waveform_averages=1, channel=0)
    plot.source = SimpleNamespace(fs=fs, calibration=calibration)
    plot._freq = freq

    # Historical batch computation (the old _y implementation).
    y = epochs[:, 0]
    if average_mode == 'time':
        y = y.mean(axis=0)
    expected = calibration.get_db(freq, util.psd(y, fs))
    if average_mode == 'FFT':
        expected = expected.mean(axis=0)

    # Incremental path: fold each epoch, render from the running mean.
    acc = EpochGroupAccumulator(transform=plot._fold)
    acc.add_epochs([_FakeEpoch(e) for e in epochs],
                   lambda md: ((), (md['level'],)))
    result = plot._render_mean(acc.get_mean(((), (60,))))

    # The DC bin of a detrended signal is numerical noise at machine
    # precision (~-300 dB); psiaudio's psd itself produces different noise
    # there for 2D vs per-row input, so it is excluded from the comparison
    # and only sanity-checked.
    np.testing.assert_allclose(result[1:], expected[1:], rtol=1e-10)
    assert result[0] < -250 and expected[0] < -250


def test_average_plot_render_mean_selects_channel():
    class _Plot(GroupedEpochAveragePlot):
        def _observe_source(self, event):
            pass

    plot = _Plot(channel=1)
    rng = np.random.RandomState(4)
    epochs = rng.normal(size=(5, 3, 20))
    acc = EpochGroupAccumulator(transform=plot._fold)
    acc.add_epochs([_FakeEpoch(e) for e in epochs],
                   lambda md: ((), (md['level'],)))
    result = plot._render_mean(acc.get_mean(((), (60,))))
    # Equivalent to the historical concat(...).mean(axis='epoch')[channel].
    np.testing.assert_allclose(result, epochs.mean(axis=0)[1], rtol=1e-12)
