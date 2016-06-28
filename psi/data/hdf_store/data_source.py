import logging
log = logging.getLogger(__name__)

import threading

from atom.api import Atom, Float, Property, Event, Typed


class DataSource(Atom):

    data = Typed(object)
    current_time = Float(0)
    added = Event()
    changed = Event()

    def set_current_time(self, current_time):
        self.current_time = current_time


class DataTable(DataSource):

    def append(self, row):
        self.data.append(row)
        self.added = row

    def query(self, string, condvars, field):
        return self.data.read_where(string, condvars, field)


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

    Two events are supported.

    added
        New data has been added. If listeners have been caching the results of
        prior computations, they can assume that older data in the cache is
        valid.
    changed
        The underlying dataset has changed, but the time-range has not.

    The changed event roughly corresponds to changes in the Y-axis (i.e. the
    signal) while added roughly corresponds to changes in the X-axis (i.e.
    addition of additional samples).
    '''
    data = Typed(object)

    # Sampling frequency of the data stored in the buffer
    fs = Float()

    # Time of first sample in the buffer.  Typically this is 0, but if we delay
    # acquisition or discard "old" data (e.g. via a RAMBuffer), then we need to
    # update t0.
    t0 = Float(0)
    shape = Property()
    current_time = Float(0)

    added = Event()
    changed = Event()

    def _get_shape(self):
        return self.data.shape

    def __getitem__(self, slice):
        '''
        Delegates to the __getitem__ method on the underlying buffer

        Subclasses can add additional data preprocessing by overriding this
        method.  See `ProcessedFileMultiChannel` for an example.
        '''
        log.trace('Reading slice {}'.format(slice))
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
                raise ValueError, "start must be >= 0"
            if ub >= len(self.data):
                raise ValueError, "end must be <= signal length"

        if np.iterable(lb):
            return [self[..., lb:ub] for lb, ub in zip(lb, ub)]
        else:
            return self[..., lb:ub]

    def get_index(self, index, reference=0):
        t0_index = int(self.t0*self.fs)
        index = max(0, index-t0_index+reference)
        return self[..., index]

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
        return self[..., lb:ub]

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

    def append(self, data):
        lb = self.get_size()
        log.trace('Reading data {}'.format(data.shape))
        self.data.append(data)
        ub = self.get_size()
        try:
            # TODO: FIXME. For some reason the chaco plots are raising an
            # error when being invalidated.
            self.added = lb/self.fs, ub/self.fs
        except:
            pass
