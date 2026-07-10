import numpy as np

from psi.context.choice import (
    ascending, counterbalanced, descending, exact_order, options,
    pseudorandom, shuffled_set,
)


def test_options_registry():
    expected = {
        'ascending', 'descending', 'pseudorandom', 'exact_order',
        'counterbalanced', 'shuffled_set',
    }
    assert set(options) == expected


def test_ascending_loops():
    gen = ascending([3, 1, 2])
    assert [next(gen) for _ in range(7)] == [1, 2, 3, 1, 2, 3, 1]


def test_ascending_stops_after_c_cycles():
    gen = ascending([1, 2, 3], c=2)
    assert list(gen) == [1, 2, 3, 1, 2, 3]


def test_ascending_with_key():
    gen = ascending(['bb', 'a', 'ccc'], c=1, key=len)
    assert list(gen) == ['a', 'bb', 'ccc']


def test_descending_loops():
    gen = descending([1, 3, 2], c=2)
    assert list(gen) == [3, 2, 1, 3, 2, 1]


def test_exact_order_preserves_order():
    gen = exact_order([5, 1, 7], c=2)
    assert list(gen) == [5, 1, 7, 5, 1, 7]


def test_shallow_copy_protects_against_external_mutation():
    seq = [1, 3, 2]
    gen = ascending(seq, c=1)
    # Mutate the source sequence; the generator must be unaffected.
    seq[0] = 99
    assert list(gen) == [1, 2, 3]


def test_pseudorandom_is_seed_reproducible():
    seq = list(range(20))
    a = [next(pseudorandom(seq, seed=42)) for _ in range(50)]
    b = [next(pseudorandom(seq, seed=42)) for _ in range(50)]
    # Each generator is independent but the same seed gives the same first draw.
    assert a[0] == b[0]
    # Compare two full sequences produced by the same seeded generator.
    g1 = pseudorandom(seq, seed=7)
    g2 = pseudorandom(seq, seed=7)
    assert [next(g1) for _ in range(50)] == [next(g2) for _ in range(50)]


def test_pseudorandom_uses_independent_state():
    # Drawing from numpy's global state must not perturb the generator.
    g = pseudorandom([1, 2, 3, 4, 5], seed=0)
    first = next(g)
    np.random.seed(123)
    np.random.random(1000)
    g2 = pseudorandom([1, 2, 3, 4, 5], seed=0)
    assert next(g2) == first


def test_shuffled_set_visits_each_value_once_per_cycle():
    seq = [10, 20, 30, 40]
    gen = shuffled_set(seq, c=3, seed=1)
    # Each cycle (4 draws) must be a permutation of the full sequence.
    for _ in range(3):
        cycle = [next(gen) for _ in range(len(seq))]
        assert sorted(cycle) == sorted(seq)


def test_counterbalanced_distribution():
    gen = counterbalanced([0, 1, 2], 60)
    items = [next(gen) for _ in range(60)]
    assert np.bincount(items).tolist() == [20, 20, 20]


def test_counterbalanced_reduces_to_shuffled_set_at_n_equals_one():
    # When n == 1, full_sequence has exactly one slot per call to np.array_split
    # over len(sequence) parts -- which actually means one draw per cycle.
    seq = ['A', 'B', 'C']
    gen = counterbalanced(seq, 3)
    cycle = [next(gen) for _ in range(3)]
    assert sorted(cycle) == sorted(seq)


def test_empty_sequence_warns_but_does_not_raise(caplog):
    # The check_sequence decorator warns; the generator just terminates.
    import logging
    with caplog.at_level(logging.WARNING):
        gen = ascending([], c=1)
        assert list(gen) == []
    assert any('Empty sequence' in r.message for r in caplog.records)
