import logging
log = logging.getLogger(__name__)

import itertools
import copy
import uuid
from collections import deque

import numpy as np

from enaml.core.api import Declarative


class QueueEmptyError(Exception):
    pass


def as_iterator(x):
    if x is None:
        x = 0
    try:
        x = iter(x)
    except TypeError:
        x = itertools.cycle([x])
    return x


class AbstractSignalQueue:

    def __init__(self, fs=None):
        '''
        Parameters
        ----------
        fs : float
            Sampling rate of output that will be using this queue
        '''
        # Used internally to track intertrial silent period.
        self._delay_samples = 0

        # Dictionary of generators or arrays. Each token added to the queue has
        # a unique ID. The token is associated with either a class-based
        # generator (which can be restarted to generate a new waveform) or an
        # already-generated waveform.
        self._data = {}

        # Tracks order of items added to queue. Subclasses will incorporate
        # this into their algorithms to determine the actual ordering of the
        # stimuli (e.g., first-in, first-out, interleaved, etc.).
        self._ordering = []

        # Current waveform generator for trials.
        self._source = None

        self._samples = 0

        # List of callbacks each time a new waveform is started.
        self._notifiers = []

        # Is stimulus generation paused?
        self._paused = False

        # Is queue complete?
        self._empty = False

        # Sampling rate needed to generate waveforms at. This is required since
        # it's used in some calculations of timing.
        self._fs = fs

        # Start time of queue relative to acquisition start.
        self._t0 = 0

    def pause(self):
        log.debug('Pausing queue')
        self._paused = True

    def resume(self):
        log.debug('Resuming queue')
        self._paused = False

    def toggle_pause(self):
        self._paused = not self._paused

    def is_empty(self):
        return self._empty

    def set_fs(self, fs):
        # Sampling rate at which samples will be generated.
        self._fs = fs

    def set_t0(self, t0):
        # Sample at which queue was started relative to experiment acquisition
        # start.
        self._t0 = t0

    def _add_source(self, source, trials, delays, duration, metadata):
        key = uuid.uuid4()
        if duration is None:
            if isinstance(source, np.ndarray):
                duration = source.shape[-1]/self._fs
            else:
                duration = source.get_duration()

        data = {
            'source': source,
            'trials': trials,
            'delays': as_iterator(delays),
            'duration': duration,
            'metadata': metadata,
        }
        self._data[key] = data
        return key

    def get_max_duration(self):
        def get_duration(source):
            try:
                return source.get_duration()
            except AttributeError:
                return source.shape[-1]/self._fs
        return max(get_duration(d['source']) for d in self._data.values())

    def connect(self, callback):
        self._notifiers.append(callback)

    def _notify(self, trial_info):
        for notifier in self._notifiers:
            notifier(trial_info)

    def insert(self, source, trials, delays=None, duration=None, metadata=None):
        k = self._add_source(source, trials, delays, duration, metadata)
        self._ordering.insert(k)
        return k

    def append(self, source, trials, delays=None, duration=None, metadata=None):
        k = self._add_source(source, trials, delays, duration, metadata)
        self._ordering.append(k)
        return k

    def count_factories(self):
        return len(self._ordering)

    def count_trials(self):
        return sum(v['trials'] for v in self._data.values())

    def is_empty(self):
        return self.count_trials() == 0

    def next_key(self):
        raise NotImplementedError

    def pop_next(self, decrement=True):
        key = self.next_key()
        return key, self.pop_key(key, decrement=decrement)

    def pop_key(self, key, decrement=True):
        '''
        Removes one trial of specified key from queue and returns waveform
        '''
        data = self._data[key]
        if decrement:
            self.decrement_key(key)
        return data

    def remove_key(self, key):
        '''
        Removes key from queue entirely, regardless of number of trials
        '''
        self._data.pop(key)
        self._ordering.remove(key)

    def decrement_key(self, key, n=1):
        if key not in self._ordering:
            raise KeyError('{} not in queue'.format(key))
        self._data[key]['trials'] -= n
        if self._data[key]['trials'] <= 0:
            self.remove_key(key)

    def _get_samples_waveform(self, samples):
        if samples > len(self._source):
            waveform = self._source
            self._source = None
        else:
            waveform = self._source[:samples]
            self._source = self._source[samples:]
        return waveform

    def _get_samples_generator(self, samples):
        samples = min(self._source.get_remaining_samples(), samples)
        waveform = self._source.next(samples)
        if self._source.is_complete():
            self._source = None
        return waveform

    def next_trial(self, decrement=True):
        '''
        Setup the next trial

        This has immediate effect. If you call this (from external code), the
        current trial will not finish.
        '''
        key, data = self.pop_next(decrement=decrement)

        self._source = data['source']
        try:
            self._source.reset()
            self._get_samples = self._get_samples_generator
        except AttributeError:
            self._source = data['source']
            self._get_samples = self._get_samples_waveform

        delay = next(data['delays'])
        self._delay_samples = int(delay*self._fs)
        if self._delay_samples < 0:
            raise ValueError('Invalid option for delay samples')

        queue_t0 = self._samples/self._fs
        uploaded = {
            't0': self._t0 + queue_t0,      # Time re. acq. start
            'queue_t0': queue_t0,           # Time re. queue start
            'duration': data['duration'],   # Duration of token
            'key': key,                     # Unique ID
            'metadata': data['metadata'],   # Metadata re. token
        }
        #log.debug('t0: %f; queue_t0: %f', uploaded['t0'], uploaded['queue_t0'])
        self._notify(uploaded)

    def pop_buffer(self, samples, decrement=True):
        '''
        Return the requested number of samples

        Removes stack of waveforms in order determind by `pop`, but only
        returns requested number of samples.  If a partial fragment of a
        waveform is returned, the remaining part will be returned on subsequent
        calls to this function.
        '''
        waveforms = []
        while samples > 0:
            try:
                waveform = self._pop_buffer(samples, decrement)
            except QueueEmptyError:
                log.info('Queue is empty')
                waveform = np.zeros(samples)
                self._empty = True
            samples -= len(waveform)
            self._samples += len(waveform)
            waveforms.append(waveform)
        waveform = np.concatenate(waveforms, axis=-1)
        log.debug('Generated %d samples', len(waveform))
        return waveform

    def _pop_buffer(self, samples, decrement):
        '''
        Encodes logic for deciding what segment needs to be generated. It must
        return *up to* the number of samples requested, but can be less if
        needed.
        '''
        # If paused, return a stream of zeros.
        if self._paused:
            return np.zeros(samples)

        # Insert intertrial interval delay if one exists
        if self._delay_samples > 0:
            n = min(self._delay_samples, samples)
            self._delay_samples -= n
            return np.zeros(n)

        # Set up next trial
        if self._source is None:
            self.next_trial(decrement)
            return np.empty(0)

        # Load samples from current source
        if self._source is not None:
            return self._get_samples(samples)


class FIFOSignalQueue(AbstractSignalQueue):
    '''
    Return waveforms based on the order they were added to the queue
    '''

    def next_key(self):
        if len(self._ordering) == 0:
            raise QueueEmptyError
        return self._ordering[0]


class InterleavedFIFOSignalQueue(AbstractSignalQueue):
    '''
    Return waveforms based on the order they were added to the queue; however,
    trials are interleaved.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._i = -1
        self._complete = False

    def next_key(self):
        if self._complete:
            raise QueueEmptyError
        self._i = (self._i + 1) % len(self._ordering)
        return self._ordering[self._i]

    def decrement_key(self, key, n=1):
        if key not in self._ordering:
            raise KeyError('{} not in queue'.format(key))
        self._data[key]['trials'] -= n
        for key, data in self._data.items():
            if data['trials'] > 0:
                return
        self._complete = True

    def count_trials(self):
        return sum(max(v['trials'], 0) for v in self._data.values())


class RandomSignalQueue(AbstractSignalQueue):
    '''
    Return waveforms in random order
    '''

    def next_key(self):
        if len(self._ordering) == 0:
            raise QueueEmptyError
        i = np.random.randint(0, len(self._ordering))
        return self._ordering[i]


class BlockedRandomSignalQueue(InterleavedFIFOSignalQueue):

    def __init__(self, seed=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._i = []
        self._rng = np.random.RandomState(seed)

    def next_key(self):
        if self._complete:
            raise QueueEmptyError
        if not self._i:
            # The blocked order is empty. Create a new set of random indices.
            i = np.arange(len(self._ordering))
            self._rng.shuffle(i)
            self._i = i.tolist()
        i = self._i.pop()
        return self._ordering[i]


class GroupedFIFOSignalQueue(FIFOSignalQueue):
    '''
    Like the FIFOSignalQueue, this queue iterates through each waveform in the
    order it was added. However, the iteration is performed in blocks. If the
    block size is 4 and you have 8 waveforms queued:

        A B C D E F G H

    The queue iterates through A B C D until all trials have been presented,
    then it shifts to E F G H.
    '''

    def __init__(self, group_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._i = -1
        self._group_size = group_size

    def next_key(self):
        if len(self._ordering) == 0:
            raise QueueEmptyError
        self._i = (self._i + 1) % self._group_size
        return self._ordering[self._i]

    def decrement_key(self, key, n=1):
        if key not in self._ordering:
            raise KeyError('{} not in queue'.format(key))
        self._data[key]['trials'] -= n

        # Check to see if the group is complete. Return from method if not
        # complete.
        for key in self._ordering[:self._group_size]:
            if self._data[key]['trials'] > 0:
                return

        # If complete, remove the keys
        for key in self._ordering[:self._group_size]:
            self.remove_key(key)


queues = {
    'first-in, first-out': FIFOSignalQueue,
    'interleaved first-in, first-out': InterleavedFIFOSignalQueue,
    'random': RandomSignalQueue,
}
