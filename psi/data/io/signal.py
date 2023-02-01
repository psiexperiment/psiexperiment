import logging
log = logging.getLogger(__name__)

import functools

import numpy as np
import pandas as pd
from scipy import signal

from psiaudio import calibration


def get_unique_columns(df, exclude=None):
    return [c for c in df if (len(df[c].unique()) > 1) and (c not in exclude)]


def find_object(node, obj_id):
    if isinstance(obj_id, str):
        obj_id = int(obj_id.split('::')[1])
    if hasattr(node, 'items'):
        if node.get('__id__') == obj_id:
            return node
        for k, v in node.items():
            try:
                return find_object(v, obj_id)
            except KeyError:
                continue
    if isinstance(node, list):
        for i in node:
            try:
                return find_object(i, obj_id)
            except KeyError:
                continue
    raise KeyError('No such node')


class Signal:


    def get_calibration(self):
        cal = self.array.attrs['source']['calibration']
        if isinstance(cal, str):
            cal = find_object(self.array.attrs, cal)
        freq = cal['frequency']
        sens = cal['sensitivity']
        return calibration.InterpCalibration(freq, sens)

    def get_epochs(self, md, offset, duration, detrend=None, downsample=None,
                   columns='auto', cb=None):
        fn = self.get_segments
        return self._get_epochs(fn, md, offset, duration, detrend=detrend,
                                downsample=downsample, columns=columns, cb=cb)

    def get_epochs_filtered(self, md, offset, duration, filter_lb, filter_ub,
                            filter_order=1, detrend='constant',
                            pad_duration=10e-3, downsample=None,
                            columns='auto', cb=None):
        fn = self.get_segments_filtered
        return self._get_epochs(fn, md=md, offset=offset, duration=duration,
                                filter_lb=filter_lb, filter_ub=filter_ub,
                                filter_order=filter_order, detrend=detrend,
                                pad_duration=pad_duration,
                                downsample=downsample, columns=columns, cb=cb)

    def _get_epochs(self, fn, md, *args, columns='auto', **kwargs):
        if columns == 'auto':
            columns = get_unique_columns(md, exclude=['t0', 'key'])
        t0 = md['t0'].values
        arrays = [md[c] for c in columns]
        arrays.append(t0)
        df = fn(t0, *args, **kwargs)
        df.index = pd.MultiIndex.from_arrays(arrays, names=columns + ['t0'])
        return df

    def get_segments(self, times, offset, duration, channel=0, detrend=None,
                     downsample=None, cb=None, cb_n=1000, allow_partial=False):
        if cb is None:
            cb = lambda *a, **kw: None

        times = np.asarray(times)
        indices = np.round((times + offset) * self.fs).astype('i')
        samples = round(duration * self.fs)

        if not allow_partial:
            m = (indices >= 0) & ((indices + samples) < self.shape[-1])
            if not m.all():
                i = np.flatnonzero(~m)
                missing = ', '.join(str(e) for e in i)
                log.warn('Missing epochs %s', missing)
            indices = indices[m]
            index = pd.Index(times[m], name='t0')
        else:
            index = pd.Index(times, name='t0')

        values = []
        n = len(indices)
        for j, i in enumerate(indices):
            # This hack is in-place to handle legacy data that was stored in 1D
            # format rather than the newer 2D format.
            if len(self.shape) == 1:
                if channel != 0:
                    raise ValueError(f'Data is 1D. Cannot load channel {channel}.')
                v = self[i:i+samples]
            else:
                v = self[channel, i:i+samples]
            pad_n = samples - v.shape[-1]
            if pad_n:
                if v.ndim == 2:
                    padding = [(0, 0), (0, pad_n)]
                elif v.ndim == 1:
                    padding = (0, pad_n)
                else:
                    raise ValueError('Unsupported dimensionality for signal')
                v = np.pad(v, padding, mode='constant', constant_values=np.nan)
            values.append(v[np.newaxis])
            if ((j+1) % cb_n) == 0:
                cb((j+1)/n)

        cb(1)

        # We need to ensure that data is cast to double since there are some
        # rare edge-cases in which precision is lost when filtering and
        # downsampling.
        values = np.concatenate(values, axis=0).astype('double')

        if detrend is not None:
            values = signal.detrend(values, axis=-1, type=detrend)

        if downsample is not None:
            values = signal.decimate(values, downsample, axis=-1)
            fs = self.fs / downsample
            samples = values.shape[-1]
        else:
            fs = self.fs

        t = np.arange(samples)/fs + offset
        columns = pd.Index(t, name='time')
        df = pd.DataFrame(values, index=index, columns=columns)
        return df.reindex(times)

    def _get_segments_filtered(self, fn, offset, duration, filter_lb,
                               filter_ub, filter_order=1, detrend='constant',
                               pad_duration=10e-3, downsample=None, cb=None):

        fs = self.fs if downsample is None else self.fs / downsample
        Wn = (filter_lb/(0.5*fs), filter_ub/(0.5*fs))
        b, a = signal.iirfilter(filter_order, Wn, btype='band', ftype='butter')
        df = fn(offset-pad_duration, duration+pad_duration, detrend=detrend,
                downsample=downsample, cb=cb)

        # Attempting to write values *back* into the original df (e.g., via
        # df[:] = result) can take up quite a bit of memory for some bizzare
        # reason (possibly a bug in a specific version of Pandas). To get
        # around this limitation, we should just create a new dataframe with
        # the same index and columns.
        filt = signal.filtfilt(b, a, df.values, axis=-1)
        df_filt = pd.DataFrame(filt, index=df.index, columns=df.columns)
        return df_filt.loc[:, offset:offset+duration]

    def get_random_segments(self, n, offset, duration, detrend, downsample):
        t_min = -offset
        t_max = self.duration-duration-offset
        times = np.random.uniform(t_min, t_max, size=n)
        return self.get_segments(times, offset, duration, detrend,
                                 downsample=downsample)

    def get_segments_filtered(self, times, *args, **kwargs):
        fn = functools.partial(self.get_segments, times)
        return self._get_segments_filtered(fn, *args, **kwargs)

    def get_random_segments_filtered(self, n, *args, **kwargs):
        fn = functools.partial(self.get_random_segments, n)
        return self._get_segments_filtered(fn, *args, **kwargs)
