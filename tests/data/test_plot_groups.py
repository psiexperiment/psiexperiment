"""Tests for psi.data.plot_groups (incremental grouped-epoch averaging)."""
import numpy as np

from psi.data.plot_groups import EpochGroupAccumulator, RunningMean


class _FakeEpoch:
    """Minimal stand-in for a PipelineData epoch: array-like with metadata."""

    def __init__(self, level, data):
        self._data = np.atleast_2d(np.asarray(data, dtype=np.double))
        self.metadata = {'level': level}
        self.shape = self._data.shape

    def __array__(self, dtype=None, copy=None):
        return self._data if dtype is None else self._data.astype(dtype)


def _epoch(level, n=10, value=1.0):
    return _FakeEpoch(level, np.full((1, n), value))


def _key(md):
    # tab key, plot key
    return (), (md['level'],)


# -------- RunningMean --------

def test_running_mean_empty():
    assert RunningMean().mean is None


def test_running_mean_matches_batch():
    rng = np.random.RandomState(0)
    epochs = rng.normal(size=(20, 3, 50))
    rm = RunningMean()
    for e in epochs:
        rm.add(e)
    np.testing.assert_allclose(rm.mean, epochs.mean(axis=0), rtol=1e-12)


def test_running_mean_ignores_nan():
    rm = RunningMean()
    rm.add([1.0, np.nan])
    rm.add([3.0, 4.0])
    np.testing.assert_array_equal(rm.mean, [2.0, 4.0])


def test_running_mean_ragged_epochs():
    rm = RunningMean()
    rm.add([1.0, 2.0])
    rm.add([3.0, 4.0, 5.0])
    # Trailing sample covered by only one epoch.
    np.testing.assert_array_equal(rm.mean, [2.0, 3.0, 5.0])


def test_running_mean_all_nan_position():
    rm = RunningMean()
    rm.add([1.0, np.nan])
    rm.add([3.0, np.nan])
    result = rm.mean
    assert result[0] == 2.0
    assert np.isnan(result[1])


# -------- EpochGroupAccumulator --------

def test_add_epochs_groups_by_key():
    acc = EpochGroupAccumulator()
    acc.add_epochs([_epoch(60), _epoch(60), _epoch(80)], _key)
    assert acc.count(((), (60,))) == 2
    assert acc.count(((), (80,))) == 1
    assert acc.count(((), (40,))) == 0
    assert acc.get_mean(((), (40,))) is None


def test_incremental_mean_matches_batch_mean():
    rng = np.random.RandomState(1)
    data = rng.normal(size=(10, 2, 30))
    acc = EpochGroupAccumulator()
    acc.add_epochs([_FakeEpoch(60, d) for d in data], _key)
    mean = acc.get_mean(((), (60,)))
    np.testing.assert_allclose(mean, data.mean(axis=0), rtol=1e-12)


def test_transform_applied_per_epoch():
    # Folding through a per-epoch transform must equal transforming each
    # epoch then averaging (used by dB-PSD averaging in FFT plots).
    transform = lambda d: np.asarray(d) ** 2
    data = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    acc = EpochGroupAccumulator(transform=transform)
    acc.add_epochs([_FakeEpoch(60, d) for d in data], _key)
    expected = (data ** 2).mean(axis=0)
    np.testing.assert_array_equal(acc.get_mean(((), (60,))), expected)


def test_add_epochs_returns_last_key():
    acc = EpochGroupAccumulator()
    key = acc.add_epochs([_epoch(60), _epoch(80)], _key)
    assert key == ((), (80,))


def test_add_epochs_empty_returns_none():
    # Regression: the caller used to reference an undefined loop variable
    # when the epoch list was empty.
    acc = EpochGroupAccumulator()
    assert acc.add_epochs([], _key) is None


def test_add_epochs_excludes_none_keys():
    acc = EpochGroupAccumulator()
    exclude_60 = lambda md: None if md['level'] == 60 else _key(md)
    acc.add_epochs([_epoch(60), _epoch(80)], exclude_60)
    assert acc.count(((), (60,))) == 0
    assert acc.count(((), (80,))) == 1


def test_needs_update_batching():
    acc = EpochGroupAccumulator(n_update=3)
    key = ((), (60,))
    acc.add_epochs([_epoch(60)], _key)
    assert not acc.needs_update(key)
    acc.add_epochs([_epoch(60), _epoch(60)], _key)
    assert acc.needs_update(key)
    acc.mark_updated(key)
    assert not acc.needs_update(key)
    # Two more is not enough; the third crosses the threshold again.
    acc.add_epochs([_epoch(60), _epoch(60)], _key)
    assert not acc.needs_update(key)
    acc.add_epochs([_epoch(60)], _key)
    assert acc.needs_update(key)


def test_tab_filtering():
    acc = EpochGroupAccumulator()
    tab_key = lambda md: (('a',) if md['level'] < 70 else ('b',), (md['level'],))
    acc.add_epochs([_epoch(60), _epoch(80)], tab_key)
    assert acc.keys_for_tab(('a',)) == [(('a',), (60,))]
    assert acc.tab_needs_update(('a',))
    acc.mark_updated((('a',), (60,)))
    assert not acc.tab_needs_update(('a',))
    assert acc.tab_needs_update(('b',))


def test_max_samples():
    acc = EpochGroupAccumulator()
    assert acc.max_samples == 0
    acc.add_epochs([_epoch(60, n=10), _epoch(80, n=25)], _key)
    assert acc.max_samples == 25


def test_reset():
    acc = EpochGroupAccumulator(n_update=2)
    acc.add_epochs([_epoch(60), _epoch(60)], _key)
    acc.reset()
    assert acc.count(((), (60,))) == 0
    assert acc.get_mean(((), (60,))) is None
    assert acc.max_samples == 0
    assert not acc.needs_update(((), (60,)))
    # n_update setting survives a reset.
    assert acc.n_update == 2


# -------- equivalence of the FFT plot's incremental dB averaging --------

def test_db_psd_calibration_is_additive():
    """The FFT plot folds db(psd(epoch)) and applies calibration to the
    mean; this must equal the old batch formula get_db(f, psd).mean(0)."""
    from psiaudio import util
    from psiaudio.calibration import FlatCalibration

    fs = 1000
    rng = np.random.RandomState(2)
    epochs = rng.normal(size=(8, 100))
    freq = np.fft.rfftfreq(100, 1 / fs)
    cal = FlatCalibration.from_db(20)

    batch = cal.get_db(freq, util.psd(epochs, fs)).mean(axis=0)
    incremental = np.mean([util.db(util.psd(e, fs)) for e in epochs], axis=0)
    incremental = incremental + cal.get_db(freq, 1)
    np.testing.assert_allclose(incremental, batch, rtol=1e-10)
