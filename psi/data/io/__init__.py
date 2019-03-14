import functools
from pathlib import Path

import numpy as np
import pandas as pd


class Recording:

    def __init__(self, base_path):
        bp = Path(base_path)
        self.base_path = bp
        self.carray_names = {d.parent.stem for d in bp.glob('*/meta')}
        self.ctable_names = {d.parent.parent.stem for d in bp.glob('*/*/meta')}
        self.ttable_names = {d.stem for d in bp.glob('*.csv')}

    def __getattr__(self, attr):
        if attr in self.carray_names:
            return self._load_bcolz_signal(attr)
        if attr in self.ctable_names:
            return self._load_bcolz_table(attr)
        elif attr in self.ttable_names:
            return self._load_text_table(attr)

    def __repr__(self):
        n_signals = len(self.carray_names)
        n_tables = len(self.ctable_names)
        return f'<Dataset with {n_signals} signals and {n_tables} tables>'

    @functools.lru_cache()
    def _load_bcolz_signal(self, name):
        from .bcolz_tools import BcolzSignal
        return BcolzSignal(self.base_path / name)

    @functools.lru_cache()
    def _load_bcolz_table(self, name):
        from .bcolz_tools import load_ctable_as_df
        return load_ctable_as_df(self.base_path / name)

    @functools.lru_cache()
    def _load_text_table(self, name):
        import pandas as pd
        return pd.read_csv((self.base_path / name).with_suffix('.csv'))


class Signal:

    def get_epochs(self, md, offset, duration, detrend=None, columns='auto'):
        fn = self.get_segments
        return self._get_epochs(fn, md, offset, duration, columns=columns)

    def get_epochs_filtered(self, md, offset, duration, filter_lb, filter_ub,
                            filter_order=1, detrend='constant',
                            pad_duration=10e-3, columns='auto'):
        fn = self.get_segments_filtered
        return self._get_epochs(fn, md, offset, duration, filter_lb, filter_ub,
                                filter_order, detrend, pad_duration,
                                columns=columns)

    def _get_epochs(self, fn, md, *args, columns='auto', **kwargs):
        if columns == 'auto':
            columns = get_unique_columns(md, exclude=['t0'])
        t0 = md['t0'].values
        arrays = [md[c] for c in columns]
        arrays.append(t0)
        df = fn(t0, *args, **kwargs)
        df.index = pd.MultiIndex.from_arrays(arrays, names=columns + ['t0'])
        return df

    def get_segments(self, times, offset, duration, detrend=None):
        times = np.asarray(times)
        indices = np.round((times + offset) * self.fs).astype('i')
        samples = round(duration * self.fs)

        m = (indices >= 0) & ((indices + samples) < self.shape[-1])
        if not m.all():
            i = np.flatnonzero(~m)
            log.warn('Missing epochs %d', i)

        values = np.concatenate([self[i:i+samples][np.newaxis] \
                                 for i in indices[m]])
        if detrend is not None:
            values = signal.detrend(values, axis=-1, type=detrend)

        t = np.arange(samples)/self.fs + offset
        columns = pd.Index(t, name='time')
        index = pd.Index(times[m], name='t0')
        df = pd.DataFrame(values, index=index, columns=columns)
        return df.reindex(times)

    def _get_segments_filtered(self, fn, offset, duration, filter_lb, filter_ub,
                               filter_order=1, detrend='constant',
                               pad_duration=10e-3):

        Wn = (filter_lb/self.fs, filter_ub/self.fs)
        b, a = signal.iirfilter(filter_order, Wn, btype='band', ftype='butter')
        df = fn(offset-pad_duration, duration+pad_duration, detrend)
        df[:] = signal.filtfilt(b, a, df.values, axis=-1)
        return df.loc[:, offset:offset+duration]

    def get_random_segments(self, n, offset, duration, detrend):
        t_min = -offset
        t_max = self.duration-duration-offset
        times = np.random.uniform(t_min, t_max, size=n)
        return self.get_segments(times, offset, duration, detrend)

    def get_segments_filtered(self, times, *args, **kwargs):
        fn = functools.partial(self.get_segments, times)
        return self._get_segments_filtered(fn, *args, **kwargs)

    def get_random_segments_filtered(self, n, *args, **kwargs):
        fn = functools.partial(self.get_random_segments, n)
        return self._get_segments_filtered(fn, *args, **kwargs)
