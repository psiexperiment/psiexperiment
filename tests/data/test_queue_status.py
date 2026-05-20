"""Tests for QueueStatus sink (psi.data.sinks.queue_status)."""
import pytest

from psiaudio.queue import AbstractSignalQueue

from psi.data.sinks.api import QueueStatus


class _FakeQueue(AbstractSignalQueue):
    """Minimal AbstractSignalQueue subclass for typed-slot acceptance."""
    def __init__(self, requested, remaining):
        # Skip super().__init__() — we only need the type identity, not the
        # real queue machinery.
        self._requested = requested
        self._remaining = remaining
        self.callbacks = {'added': [], 'decrement': [], 'empty': [],
                          'removed': []}

    def connect(self, cb, event):
        self.callbacks[event].append(cb)

    def count_requested_trials(self):
        return self._requested

    def count_trials(self):
        return self._remaining


@pytest.fixture(autouse=True)
def synchronous_deferred_call(monkeypatch):
    monkeypatch.setattr(
        'psi.data.sinks.queue_status.deferred_call',
        lambda fn, *args, **kwargs: fn(*args, **kwargs),
    )


def test_queue_status_update_requested_n():
    qs = QueueStatus()
    qs.queue = _FakeQueue(requested=10, remaining=4)
    qs._update_requested_n()
    assert qs.requested_n == 10
    # current_n = requested(10) - remaining(4)
    assert qs.current_n == 6


def test_queue_status_update_current_n_recomputes():
    qs = QueueStatus()
    qs.queue = _FakeQueue(requested=10, remaining=2)
    qs.requested_n = 10
    qs._update_current_n()
    assert qs.current_n == 8


def test_queue_status_observe_queue_connects_callbacks():
    qs = QueueStatus()
    fake = _FakeQueue(requested=5, remaining=5)
    qs.queue = fake
    # Setting queue should have wired up the 'added' and 'decrement' subscribers.
    assert len(fake.callbacks['added']) == 1
    assert len(fake.callbacks['decrement']) == 1
