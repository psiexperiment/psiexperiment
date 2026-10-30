import pandas as pd

from psi.data.sinks.api import TableStore


def test_table_create_append(tmpdir):
    store = TableStore(name='event_log')
    store.set_base_path(tmpdir, is_temp=False)
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


def test_row_during_prepare_lands_after_header(tmpdir):
    # A row that arrives from another thread while `prepare` is writing the
    # rows queued before it (e.g., an event logged from the GUI thread) must
    # not be written ahead of the header.
    import threading

    threads = []

    class Store(TableStore):
        def _save_data(self, data):
            if not threads and self._store is not None:
                # Deliver a row from another thread while `prepare` is
                # partway through, and give it time to get written.
                late = threading.Thread(target=self.process_table,
                                        args=({'event': 'late'},))
                threads.append(late)
                late.start()
                late.join(0.2)
            super()._save_data(data)

    store = Store(name='event_log')
    store.set_base_path(tmpdir, is_temp=False)
    store.process_table({'event': 'early'})
    store.prepare()
    threads[0].join()
    store.flush()
    assert list(pd.read_csv(store.get_filename()).event) == ['early', 'late']
