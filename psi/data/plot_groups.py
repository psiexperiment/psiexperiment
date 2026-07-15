'''
Grouped-epoch accumulation for the plotting subsystem.

`EpochGroupAccumulator` tracks epochs grouped by a (tab_key, plot_key) pair
and implements the `n_update` batching rule: a group's plot is only redrawn
once at least `n_update` new epochs have arrived since its last redraw.
Pure Python — no Qt — so the bookkeeping is testable without a GUI.
'''


class EpochGroupAccumulator:

    def __init__(self, n_update=1):
        self.n_update = n_update
        self.reset()

    def reset(self):
        #: Maps group key to the list of epochs acquired so far.
        self._cache = {}
        #: Maps group key to the number of epochs acquired.
        self._count = {}
        #: Maps group key to the number of epochs at the last redraw.
        self._updated = {}
        #: Maps group key to the maximum epoch length (in samples).
        self._n_samples = {}

    def add_epochs(self, epochs, group_key):
        '''
        File each epoch under group_key(epoch.metadata).

        Epochs whose key is None are excluded. Returns the key of the last
        epoch processed (which may be None), or None if `epochs` is empty.
        '''
        key = None
        for d in epochs:
            key = group_key(d.metadata)
            if key is not None:
                self._cache.setdefault(key, []).append(d)
                self._count[key] = self._count.get(key, 0) + 1
                n = max(self._n_samples.get(key, 0), d.shape[-1])
                self._n_samples[key] = n
        return key

    def get(self, key):
        return self._cache.get(key, [])

    def needs_update(self, key):
        current_n = self._count.get(key, 0)
        last_n = self._updated.get(key, 0)
        return current_n >= (last_n + self.n_update)

    def mark_updated(self, key):
        self._updated[key] = len(self.get(key))

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
