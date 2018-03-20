import logging
log = logging.getLogger(__name__)

import threading
from collections import Counter

from ..sink import Sink
import numpy as np

import pandas as pd

from atom.api import Atom, Float, Property, Event, Typed, Bool


class DataSource(Atom):

    data = Typed(object)
    current_time = Float(0)
    added = Event()

    def set_current_time(self, current_time):
        self.current_time = current_time


class DataTable(DataSource):

    def append(self, row):
        self.data.append(row)

    def query(self, string, condvars, field):
        if len(self.data) == 0:
            return []
        query = string.format(**condvars)
        subset = self.data[query]
        return subset[field]


class EventDataTable(DataTable):

    lock = Typed(object)

    def _default_lock(self):
        return threading.RLock()

    def append(self, row):
        with self.lock:
            super().append(row)
        self.added = {
            'lb': row[0],
            'ub': row[0],
            'event': row[1],
        }

    def get_epochs(self, start_event, end_event, lb, ub, current_time=np.nan):
        # We need to specify the string literal as a bytes object. This seems
        # to be an obscure edge-case of Python 3 + numexpr.
        query = 'event == b"{e}"'
        column = 'timestamp'
        with self.lock:
            starts = self.query(query, {'e': start_event}, column)
            ends = self.query(query, {'e': end_event}, column)

        if len(starts) == 0 and len(ends) == 1:
            starts = [0]
        elif len(starts) == 1 and len(ends) == 0:
            ends = [current_time]
        elif len(starts) > 0 and len(ends) > 0:
            if starts[0] > ends[0]:
                starts = np.r_[0, starts]
            if starts[-1] > ends[-1]:
                ends = np.r_[ends, current_time]

        try:
            epochs = np.c_[starts, ends]
        except:
            raise
        m = ((epochs >= lb) & (epochs < ub)) | np.isnan(epochs)
        return epochs[m.any(axis=-1)]


class DataChannel(DataSource):
    '''
    Base class for dealing with a continuous stream of data sampled at a fixed
    rate, fs (cycles per time unit), starting at time t0 (time unit).  This
    class is not meant to be used directly since it does not implement a
    backend for storing the data.

    fs
        Sampling frequency
    t0
        Time offset (i.e. time of first sample) relative to the start of
        acquisition.  This typically defaults to zero; however, some subclasses
        may discard old data (e.g.  :class:`RAMChannel`), so we need to factor
        in the time offset when attempting to extract a given segment of the
        waveform for analysis.
    added
        New data has been added. If listeners have been caching the results of
        prior computations, they can assume that older data in the cache is
        valid.
    '''
    # Sampling frequency of the data stored in the buffer
    fs = Float()

    # Time of first sample in the buffer.  Typically this is 0, but if we delay
    # acquisition or discard "old" data (e.g. via a RAMBuffer), then we need to
    # update t0.
    t0 = Float(0)
    shape = Property()

    def _get_shape(self):
        return self.data.shape

    def __getitem__(self, slice):
        '''
        Delegates to the __getitem__ method on the underlying buffer

        Subclasses can add additional data preprocessing by overriding this
        method.  See `ProcessedFileMultiChannel` for an example.
        '''
        return self.data[slice]

    def to_index(self, time):
        '''
        Convert time to the corresponding index in the waveform.  Note that the
        index may be negative if the time is less than t0.  Since Numpy allows
        negative indices, be sure to check the value.
        '''
        return int((time-self.t0)*self.fs)

    def to_samples(self, time):
        '''
        Convert time to number of samples.
        '''
        time = np.asanyarray(time)
        samples = time*self.fs
        return samples.astype('i')

    def get_range_index(self, start, end, reference=0, check_bounds=False):
        '''
        Returns a subset of the range specified in samples

        The samples must be speficied relative to start of data acquisition.

        Parameters
        ----------
        start : num samples (int)
            Start index in samples
        end : num samples (int)
            End index in samples
        reference : num samples (int), optional
            Time of trigger to reference start and end to
        check_bounds : bool
            Check that start and end fall within the valid data range
        '''
        t0_index = int(self.t0*self.fs)
        lb = start-t0_index+reference
        ub = end-t0_index+reference

        if check_bounds:
            if lb < 0:
                raise ValueError("start must be >= 0")
            if ub >= len(self.data):
                raise ValueError("end must be <= signal length")

        if np.iterable(lb):
            return [self[lb:ub] for lb, ub in zip(lb, ub)]
        else:
            return self[lb:ub]

    def get_index(self, index, reference=0):
        t0_index = int(self.t0*self.fs)
        index = max(0, index-t0_index+reference)
        return self[:, index]

    def _to_bounds(self, start, end, reference=None):
        if start > end:
            raise ValueError("Start time must be < end time")
        if reference is not None:
            ref_idx = self.to_index(reference)
        else:
            ref_idx = 0
        lb = max(0, self.to_index(start)+ref_idx)
        ub = max(0, self.to_index(end)+ref_idx)
        return lb, ub

    def get_range(self, start, end, reference=None):
        '''
        Returns a subset of the range.

        Parameters
        ----------
        start : float, sec
            Start time.
        end : float, sec
            End time.
        reference : float, optional
            Set to -1 to get the most recent range
        '''
        lb, ub = self._to_bounds(start, end, reference)
        return self[lb:ub]

    def get_size(self):
        return self.data.shape[-1]

    def get_bounds(self):
        '''
        Returns valid range of times as a tuple (lb, ub)
        '''
        return self.t0, self.t0 + self.get_size()/self.fs

    def latest(self):
        if self.get_size() > 0:
            return self.data[-1]/self.fs
        else:
            return self.t0

    @property
    def n_samples(self):
        return self.data.shape[-1]


class ContinuousDataChannel(DataChannel):

    def append(self, data):
        lb = self.get_size()
        self.data.append(data)
        ub = self.get_size()
        self.added = {'ub': ub/self.fs}


class EpochDataChannel(DataChannel):

    metadata = Typed(list, [])
    lock = Typed(object)
    epoch_size = Typed(float)

    def _default_lock(self):
        return threading.Lock()

    def append(self, data):
        epochs = []
        metadata = []
        for d in data:
            epochs.append(d['signal'])
            md = d['info']['metadata'].copy()
            md['t0'] = d['info']['t0']
            md['duration'] = d['info']['duration']
            metadata.append(md)

        with self.lock:
            self.data.append(epochs)
            self.metadata.extend(metadata)
        self.added = data

    def get_epoch_groups(self, groups):
        if len(self.data) == 0:
            return {}
        metadata = pd.DataFrame(self.metadata)
        epochs = {}
        for keys, df in metadata.groupby(groups):
            i = df.index.values.astype(np.int32)
            epochs[keys] = self.data[i]
        return epochs

    def get_epochs(self, filters=None):
        if len(self.data) == 0:
            return []

        if filters is None:
            with self.lock:
                return self.data[:]

        def match(row, filters):
            for k, v in filters.items():
                if row[k] != v:
                    return False
            return True

        with self.lock:
            mask = np.array([match(m, filters) for m in self.metadata])
            return self.data[:][mask]

    def count_groups(self, grouping):
        groups = [tuple(m[g] for g in grouping) for m in self.metadata]
        return Counter(groups)

    @property
    def n_epochs(self):
        return len(self.data)


class AbstractStore(Sink):

    _stores = Typed(dict, {})

    def prepare(self, plugin):
        self._prepare_event_log()
        self._prepare_trial_log(plugin.context_info)
        # TODO: This seems a bit hackish. Do we really need this?
        self._stores['trial_log'] = self.trial_log
        self._stores['event_log'] = self.event_log

    def get_source(self, source_name):
        try:
            return self._stores[source_name]
        except KeyError as e:
            # TODO: Once we port to Python 3, add exception chaining.
            raise AttributeError(source_name) from e

    def set_current_time(self, name, timestamp):
        self._stores[name].set_current_time(timestamp)

    def _prepare_trial_log(self, context_info):
        data = self._create_trial_log(context_info)
        self.trial_log = DataTable(data=data)

    def _prepare_event_log(self):
        data = self._create_event_log()
        self.event_log = EventDataTable(data=data)
