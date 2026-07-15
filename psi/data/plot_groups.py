'''
Grouped-epoch accumulation for the plotting subsystem.

`EpochGroupAccumulator` tracks epochs grouped by a (tab_key, plot_key) pair
and implements the `n_update` batching rule: a group's plot is only redrawn
once at least `n_update` new epochs have arrived since its last redraw.

Averaging is *incremental*: each epoch is folded into a per-group running
mean as it arrives (optionally through a `transform` applied per epoch, for
plots that average in a transformed space such as dB-PSD). This keeps redraw
cost proportional to the epoch length rather than to the number of epochs
acquired so far, and means raw epochs do not need to be retained — memory is
O(groups x samples) instead of O(total epochs).

Pure Python — no Qt — so the bookkeeping is testable without a GUI.
'''
import numpy as np


class RunningMean:
    '''
    Incremental mean of arrays, ignoring NaN values.

    Arrays may grow along the last axis (ragged epochs): shorter arrays are
    treated as missing data (NaN) for the trailing samples, so the mean at
    each sample position reflects only the epochs that covered it.
    '''

    def __init__(self):
        self._sum = None
        self._count = None

    def add(self, arr):
        arr = np.asarray(arr, dtype=np.double)
        # Only NaN marks a missing sample. Infinities (e.g., dB of an empty
        # spectral bin) participate arithmetically, matching what a batch
        # mean over the same epochs would produce.
        present = ~np.isnan(arr)
        values = np.where(present, arr, 0)

        if self._sum is None:
            self._sum = values.astype(np.double)
            self._count = present.astype(np.intp)
            return

        n_old = self._sum.shape[-1]
        n_new = arr.shape[-1]
        if n_new > n_old:
            pad = [(0, 0)] * (self._sum.ndim - 1) + [(0, n_new - n_old)]
            self._sum = np.pad(self._sum, pad)
            self._count = np.pad(self._count, pad)
        self._sum[..., :n_new] += values
        self._count[..., :n_new] += present

    @property
    def mean(self):
        '''
        Current mean, or None if nothing has been added. Positions never
        covered by any epoch are NaN.
        '''
        if self._sum is None:
            return None
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self._count > 0, self._sum / self._count, np.nan)


class EpochGroupAccumulator:

    def __init__(self, n_update=1, transform=None):
        self.n_update = n_update
        #: Applied to each epoch before folding it into the group mean
        #: (e.g., dB-PSD for FFT-averaged plots). Must be a pure function of
        #: the epoch; parameters it closes over (fs, channel, ...) must not
        #: change once epochs have been folded. Defaults to np.asarray.
        self.transform = transform if transform is not None else np.asarray
        self.reset()

    def reset(self):
        #: Maps group key to the running mean of transformed epochs.
        self._means = {}
        #: Maps group key to the number of epochs acquired.
        self._count = {}
        #: Maps group key to the number of epochs at the last redraw.
        self._updated = {}
        #: Maps group key to the maximum epoch length (in samples).
        self._n_samples = {}

    def add_epochs(self, epochs, group_key):
        '''
        Fold each epoch into the running mean for group_key(epoch.metadata).

        Epochs whose key is None are excluded. Returns the key of the last
        epoch processed (which may be None), or None if `epochs` is empty.
        '''
        key = None
        for d in epochs:
            key = group_key(d.metadata)
            if key is not None:
                self._means.setdefault(key, RunningMean()) \
                    .add(self.transform(d))
                self._count[key] = self._count.get(key, 0) + 1
                n = max(self._n_samples.get(key, 0), d.shape[-1])
                self._n_samples[key] = n
        return key

    def get_mean(self, key):
        '''
        Running mean of the (transformed) epochs for this group, or None if
        the group has no epochs.
        '''
        mean = self._means.get(key)
        return None if mean is None else mean.mean

    def count(self, key):
        return self._count.get(key, 0)

    def needs_update(self, key):
        current_n = self._count.get(key, 0)
        last_n = self._updated.get(key, 0)
        return current_n >= (last_n + self.n_update)

    def mark_updated(self, key):
        self._updated[key] = self._count.get(key, 0)

    def keys_for_tab(self, tab_key):
        '''
        Return all group keys whose tab component matches tab_key.
        '''
        return [k for k in self._count if k[0] == tab_key]

    def tab_needs_update(self, tab_key):
        '''
        True if at least one group in the tab has accumulated enough new
        epochs to warrant a redraw.
        '''
        return any(self.needs_update(k) for k in self.keys_for_tab(tab_key))

    @property
    def max_samples(self):
        '''
        Longest epoch (in samples) seen across all groups (0 if no epochs).
        '''
        return max(self._n_samples.values(), default=0)
