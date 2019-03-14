import enaml
import json
import numpy as np
import pandas as pd
import random

with enaml.imports():
    from psi.data.sinks.text_store import TextStore


def test_table_create_append():
    store = TextStore()
    metadata = {'testing': True, 'tester': 'psiexperiment'}

    store.create_table('event_log', **metadata)
    rows = [
        {
            'channel': 'A',
            'fs': 100e3,
            'result': 5,
            'valid': True,
        },
        # Mix up order to make sure that we still get good output.
        {
            'valid': True,
            'result': 5,
            'channel': 'A',
            'fs': 100e3,
        },
    ]
    for row in rows:
        store.process_table('event_log', row, flush=True)

    filename = store.get_filename('event_log', '.csv')
    md_filename = store.get_filename('event_log_metadata', '.json')

    assert pd.read_csv(filename).equals(pd.DataFrame(rows))
    with md_filename.open() as fh:
        assert json.load(fh) == metadata


def _random_row():
    return {
        'channel': ''.join(random.choice('ABCabc123') for i in range(10)),
        'fs': np.random.uniform(100, 100e3),
        'result': int(np.random.uniform(-5, 5500)),
        'valid': bool(np.random.uniform(0, 2)),
    }


def test_table_append_first_row_speed(benchmark):
    store = TextStore()
    store.create_table('event_log')
    benchmark(store.process_table, 'event_log', _random_row())


def test_table_append_10k_rows_speed(benchmark):
    store = TextStore()
    store.create_table('event_log')
    data = [_random_row() for i in range(10000)]
    store.process_table('event_log', data)
    row = _random_row()
    benchmark(store.process_table, 'event_log', row)
