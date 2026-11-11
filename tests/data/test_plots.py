from types import SimpleNamespace

import numpy as np
import pytest
from atom.api import Atom, Float, List

from psiaudio import util
from psiaudio.calibration import FlatCalibration
from psiaudio.pipeline import PipelineData

from psi.data.plot_groups import EpochGroupAccumulator
from psi.data.plots import (
    EpochTimeContainer, GroupedEpochAveragePlot, GroupedEpochFFTPlot,
    StackedEpochAveragePlot, ViewBox, make_color,
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


def test_request_update_coalesces(monkeypatch):
    """request_update marshals redraws to the GUI thread via deferred_call
    and coalesces bursts into a single update per event-loop pass."""
    from psi.data import plots

    deferred = []
    monkeypatch.setattr(plots, 'deferred_call', lambda cb, *a: deferred.append(cb))

    updates = []

    class _Plot(plots.BasePlot):
        def update(self, event=None):
            updates.append(1)

    # No redraw limit here (see test_request_update_limits_rate).
    plot = _Plot(max_redraw_rate=0)

    # A burst of data callbacks posts exactly one deferred redraw.
    plot.request_update()
    plot.request_update()
    plot.request_update()
    assert len(deferred) == 1
    assert updates == []

    # The GUI thread runs the deferred callback: one redraw.
    deferred.pop()()
    assert updates == [1]

    # New data after the redraw schedules the next one.
    plot.request_update()
    assert len(deferred) == 1
    deferred.pop()()
    assert updates == [1, 1]


def test_request_update_limits_rate(monkeypatch):
    """Redraws are spaced at least 1 / max_redraw_rate apart: painting
    holds the GIL and delayed the sound card callback (the device then
    played silence)."""
    from psi.data import plots

    now = [100.0]
    deferred, timed = [], []
    monkeypatch.setattr(plots, 'perf_counter', lambda: now[0])
    monkeypatch.setattr(plots, 'deferred_call',
                        lambda cb, *a: deferred.append(cb))
    monkeypatch.setattr(plots, 'timed_call',
                        lambda ms, cb, *a: timed.append((ms, cb)))

    updates = []

    class _Plot(plots.BasePlot):
        def update(self, event=None):
            updates.append(now[0])

    # The default comes from the PSI_PLOT_MAX_REDRAW_RATE setting.
    assert _Plot().max_redraw_rate == 20
    plot = _Plot()

    # The first redraw runs as soon as the GUI thread gets to it.
    plot.request_update()
    deferred.pop()()
    assert updates == [100.0]

    # 10 ms later: waits out the rest of the 50 ms, coalescing the burst.
    now[0] = 100.01
    plot.request_update()
    plot.request_update()
    assert deferred == []
    (ms, callback), = timed
    assert 40 <= ms <= 41  # rounded up to whole ms
    now[0] = 100.05
    callback()
    assert updates == [100.0, 100.05]

    # Data arriving after the interval redraws right away.
    timed.clear()
    now[0] = 100.2
    plot.request_update()
    assert timed == [] and len(deferred) == 1


def test_running_mean_concurrent_add_and_read():
    """The GUI thread reads the mean while the acquisition thread folds
    epochs (including array growth); this must never raise."""
    import threading

    from psi.data.plot_groups import RunningMean

    rm = RunningMean()
    rm.add(np.zeros(10))
    errors = []
    stop = threading.Event()

    def writer():
        try:
            n = 10
            while not stop.is_set():
                rm.add(np.random.uniform(size=n))
                n += 1  # force periodic growth/reallocation
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            while not stop.is_set():
                mean = rm.mean
                assert mean is not None
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    import time
    time.sleep(0.2)
    stop.set()
    for t in threads:
        t.join(5)
    assert errors == []


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


def test_time_axis_labels_hide_floating_point_error(app):
    from psi.data.plots import TimeAxisItem
    axis = TimeAxisItem(fmt='{S}', orientation='bottom')
    values = np.arange(4) * 0.05
    assert axis.tickStrings(values, 1, 0.05)[3] == '0.15'
    axis = TimeAxisItem(fmt='{S}', orientation='bottom')
    assert axis.tickStrings(np.arange(4) * 0.001, 1, 0.001)[3] == '0.003'


class _FakeEpochSource(Atom):

    fs = Float(1000)
    duration = Float(0.01)
    callbacks = List()

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def send(self, data):
        for cb in self.callbacks:
            cb(data)


@pytest.mark.parametrize('plot_class', [GroupedEpochAveragePlot,
                                        StackedEpochAveragePlot])
def test_epoch_plot_sets_container_range(app, plot_class):
    # The time container starts at its span (fixed when it is built, before
    # the experiment sizes the epochs) and must widen to the epochs plotted.
    container = EpochTimeContainer(span=8.5e-3)
    viewbox = ViewBox(parent=container)
    plot = plot_class(parent=viewbox)
    source = _FakeEpochSource()
    plot.source = source
    source.send(PipelineData(np.zeros((1, 1, 65)), fs=1000, s0=0,
                             metadata=[{}]))
    assert container.data_range.current_range == (0, 0.065)
