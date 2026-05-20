"""Tests for the `update` helper in psi.data.sinks.event_log.

The helper writes formatted event payloads to an optional store and/or
table. We feed in lightweight fakes to inspect the behavior.
"""
import json
from types import SimpleNamespace

import enaml


with enaml.imports():
    from psi.data.sinks.event_log import update


class _FakeStore:
    def __init__(self):
        self.rows = []

    def process_table(self, row):
        self.rows.append(row)


class _FakeTable:
    def __init__(self):
        self.rows = []

    def append_row(self, row):
        self.rows.append(row)


def _event(name, info=None):
    return SimpleNamespace(parameters={'data': {
        'event': name,
        'timestamp': 1.0,
        'info': info or {},
    }})


def test_update_writes_to_store_and_table():
    store, table = _FakeStore(), _FakeTable()
    update([], store, table, _event('trial_start', {'reward': 1}))
    assert len(store.rows) == 1
    assert len(table.rows) == 1
    # The info field should be JSON-stringified.
    assert json.loads(store.rows[0]['info']) == {'reward': 1}
    assert store.rows[0]['event'] == 'trial_start'


def test_update_exclude_skips_matching_events():
    store = _FakeStore()
    update(['trial_*'], store, None, _event('trial_start'))
    update(['trial_*'], store, None, _event('iti_active'))
    assert [r['event'] for r in store.rows] == ['iti_active']


def test_update_no_store_or_table_is_silent():
    # Should not raise even with both store and table set to None.
    update([], None, None, _event('any_event'))


def test_update_handles_no_info_field():
    store = _FakeStore()
    update([], store, None, _event('e', info={}))
    assert store.rows[0]['info'] == '{}'
