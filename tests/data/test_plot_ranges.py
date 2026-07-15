"""Tests for psi.data.plot_ranges (visible-range state machines, no Qt)."""
from types import SimpleNamespace


from psi.data.plot_ranges import ChannelDataRange, EpochDataRange


class _FakeSource:

    def __init__(self, name='signal'):
        self.name = name
        self.callbacks = []

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def send(self, t_end):
        for cb in self.callbacks:
            cb(SimpleNamespace(t_end=t_end, duration=t_end))


def test_channel_range_defaults():
    r = ChannelDataRange(span=5)
    assert r.current_range == (0, 5)


def test_channel_range_advances_in_span_multiples():
    r = ChannelDataRange(span=5)
    source = _FakeSource()
    r.add_source(source)

    # Data within the first window does not move the range.
    source.send(3.0)
    assert r.current_range == (0, 5)

    # Crossing the span boundary advances the window by one span.
    source.send(5.5)
    assert r.current_range == (5, 10)

    source.send(17.2)
    assert r.current_range == (15, 20)


def test_channel_range_delay_shifts_window():
    r = ChannelDataRange(span=5, delay=1)
    source = _FakeSource()
    r.add_source(source)
    source.send(5.5)
    assert r.current_range == (4, 9)


def test_channel_range_time_never_regresses():
    r = ChannelDataRange(span=5)
    source = _FakeSource()
    r.add_source(source)
    source.send(12.0)
    source.send(3.0)  # Late/out-of-order data must not rewind the window.
    assert r.current_time == 12.0
    assert r.current_range == (10, 15)


def test_channel_range_span_change_updates_range():
    r = ChannelDataRange(span=5)
    source = _FakeSource()
    r.add_source(source)
    source.send(12.0)
    r.span = 20
    assert r.current_range == (0, 20)


def test_channel_range_track_sources_filters():
    r = ChannelDataRange(span=5, track_sources=['tracked'])
    tracked = _FakeSource('tracked')
    ignored = _FakeSource('ignored')
    r.add_source(tracked)
    r.add_source(ignored)
    assert tracked.callbacks and not ignored.callbacks


def test_epoch_range_tracks_max_duration():
    r = EpochDataRange()
    source = _FakeSource()
    r.add_source(source)
    source.send(0.5)
    assert r.current_range == (0, 0.5)
    source.send(2.0)
    assert r.current_range == (0, 2.0)
    # Shorter epochs do not shrink the range.
    source.send(1.0)
    assert r.current_range == (0, 2.0)
