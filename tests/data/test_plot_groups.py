"""Tests for psi.data.plot_groups.EpochGroupAccumulator."""
from types import SimpleNamespace


from psi.data.plot_groups import EpochGroupAccumulator


def _epoch(level, n=10):
    return SimpleNamespace(metadata={'level': level},
                           shape=(1, n))


def _key(md):
    # tab key, plot key
    return (), (md['level'],)


def test_add_epochs_groups_by_key():
    acc = EpochGroupAccumulator()
    acc.add_epochs([_epoch(60), _epoch(60), _epoch(80)], _key)
    assert len(acc.get(((), (60,)))) == 2
    assert len(acc.get(((), (80,)))) == 1
    assert acc.get(((), (40,))) == []


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
    assert acc.get(((), (60,))) == []
    assert len(acc.get(((), (80,)))) == 1


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
    assert acc.get(((), (60,))) == []
    assert acc.max_samples == 0
    assert not acc.needs_update(((), (60,)))
    # n_update setting survives a reset.
    assert acc.n_update == 2
