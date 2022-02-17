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
