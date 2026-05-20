"""Tests for the epoch-counter sinks.

The implementations use `enaml.application.deferred_call` to update Atom
members. We monkeypatch that to run synchronously so we can assert on the
counter state directly.
"""
import numpy as np
import pytest

from psi.data.sinks.api import EpochCounter, GroupedEpochCounter, SimpleCounter


@pytest.fixture(autouse=True)
def synchronous_deferred_call(monkeypatch):
    """Run any deferred_call(fn, *args) synchronously inside the test."""
    monkeypatch.setattr(
        'psi.data.sinks.epoch_counter.deferred_call',
        lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )


class _Epoch:
    """Bare-bones stand-in for the psiaudio Epoch object."""
    def __init__(self, metadata):
        self.metadata = metadata


class _EpochArray(np.ndarray):
    """ndarray subclass with `.metadata` attached per element-like access.

    EpochCounter._update_data uses ``epochs.shape[0]``. GroupedEpochCounter
    iterates and reads ``e.metadata`` per epoch. We satisfy both shapes with
    a list of _Epoch instances for grouped and a numpy array for simple.
    """


def test_simple_counter_default_state():
    sc = SimpleCounter()
    assert sc.current_n == 0
    assert sc.requested_n == 0


def test_epoch_counter_increments_current_n():
    ec = EpochCounter(name='ec', output_name='dummy')
    # ``epochs`` only needs a `.shape[0]` for EpochCounter's _update_data.
    epochs = np.zeros((3, 100))
    ec._update_data(epochs)
    assert ec.current_n == 3
    ec._update_data(np.zeros((2, 100)))
    assert ec.current_n == 5


def test_grouped_epoch_counter_counts_per_key():
    # Bypass the ContextMeta-typed `groups` member by setting get_key directly.
    gc = GroupedEpochCounter(name='gc', output_name='dummy')
    gc.get_key = lambda md: (md['stim'], md['level'])
    epochs = [
        _Epoch({'stim': 'A', 'level': 60}),
        _Epoch({'stim': 'A', 'level': 60}),
        _Epoch({'stim': 'B', 'level': 60}),
        _Epoch({'stim': 'A', 'level': 70}),
        _Epoch({'stim': 'B', 'level': 60}),
    ]
    gc._update_data(epochs)
    # Counts per (stim, level): (A,60)=2, (B,60)=2, (A,70)=1. None exceed
    # requested_group_n, so current_n = 2 + 2 + 1 = 5.
    assert gc.current_n == 5


def test_grouped_epoch_counter_caps_at_requested_group_n():
    gc = GroupedEpochCounter(name='gc', output_name='dummy', requested_group_n=2)
    gc.get_key = lambda md: md['stim']
    epochs = [_Epoch({'stim': 'A'}) for _ in range(5)]
    gc._update_data(epochs)
    # All 5 share one key; capped at requested_group_n.
    assert gc.current_n == 2
