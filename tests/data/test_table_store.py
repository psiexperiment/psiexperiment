import numpy as np
import pandas as pd
import random

from psi.data.sinks.api import TableStore


def test_table_create_append():
    store = TableStore(name='event_log')
    store.prepare()
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
        store.process_table(row)
        store.flush()
    filename = store.get_filename()
    assert pd.read_csv(filename).equals(pd.DataFrame(rows))


def _random_row():
    return {
        'channel': ''.join(random.choice('ABCabc123') for i in range(10)),
        'fs': np.random.uniform(100, 100e3),
        'result': int(np.random.uniform(-5, 5500)),
        'valid': bool(np.random.uniform(0, 2)),
    }


def test_table_append_first_row_speed(benchmark):
    store = TableStore(name='event_log')
    store.prepare()
    benchmark(store.process_table, _random_row())


def test_table_append_10k_rows_speed(benchmark):
    store = TableStore(name='event_log')
    store.prepare()
    data = [_random_row() for i in range(10000)]
    store.process_table(data)
    row = _random_row()
    benchmark(store.process_table, row)
