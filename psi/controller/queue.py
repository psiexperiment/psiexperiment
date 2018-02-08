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


class QueueBufferEmptyError(Exception):
    pass


def as_iterator(x):
    if x is None:
        x = 0
    try:
        x = iter(x)
    except TypeError:
        x = itertools.cycle([x])
    return x


class AbstractSignalQueue(object):

    def __init__(self, fs, initial_delay=0):
        self._fs = fs
        self._data = {} # list of generators
        self._ordering = [] # order of items added to queue
        self._source = None
        self._samples = 0
        self._notifiers = []
        self._delay_samples = int(initial_delay * fs)
        if self._delay_samples < 0:
            raise ValueError('Invalid option for initial delay')

    def _add_source(self, source, trials, delays, duration, metadata):
        key = uuid.uuid4()
        if duration is None:
            try:
                duration = source.shape[-1]/self._fs
            except AttributeError:
                pass
        data = {
            'source': source,
            'trials': trials,
            'delays': as_iterator(delays),
            'duration': duration,
            'metadata': metadata,
        }
        self._data[key] = data
        return key

    def connect(self, callback):
        self._notifiers.append(callback)

    def create_connection(self):
        queue = deque()
        self.connect(queue.append)
        return queue

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
            complete = True
        else:
            waveform = self._source[:samples]
            self._source = self._source[samples:]
            complete = False
        return waveform, complete

    def _get_samples_generator(self, samples):
        return self._source.send({'samples': samples})

    def pop_buffer(self, samples, decrement=True):
        '''
        Return the requested number of samples

        Removes stack of waveforms in order determind by `pop`, but only returns
        requested number of samples.  If a partial fragment of a waveform is
        returned, the remaining part will be returned on subsequent calls to
        this function.
        '''
        waveforms = []
        queue_empty = False

        if samples > 0 and self._source is not None:
            # That this is a dynamic function that is set when the next
            # source is loaded (see below in this method).
            waveform, complete = self._get_samples(samples)
            samples -= len(waveform)
            self._samples += len(waveform)
            waveforms.append(waveform)
            if complete:
                self._source = None

        if samples > 0 and self._delay_samples > 0:
            n_padding = min(self._delay_samples, samples)
            waveform = np.zeros(n_padding)
            samples -= n_padding
            self._samples += len(waveform)
            self._delay_samples -= n_padding
            waveforms.append(waveform)

        if (self._source is None) and (self._delay_samples == 0):
            try:
                key, data = self.pop_next(decrement=decrement)
                if callable(data['source']):
                    # Be sure to start the factory (hmm... shouldn't this be
                    # started in advance?)
                    self._source = data['source']()
                    self._get_samples = self._get_samples_generator
                    next(self._source)
                else:
                    self._source = data['source']
                    self._get_samples = self._get_samples_waveform

                delay = next(data['delays'])
                self._delay_samples = int(delay*self._fs)
                if self._delay_samples < 0:
                    raise ValueError('Invalid option for delay samples')

                t0 = self._samples/self._fs
                duration = data['duration']
                args = t0, duration, key, data['metadata']
                for cb in self._notifiers:
                    cb(args)

            except QueueEmptyError:
                queue_empty = True
                waveform = np.zeros(samples)
                waveforms.append(waveform)
                log.info('Queue is now empty')

        if (samples > 0) and not queue_empty:
            waveform, queue_empty = self.pop_buffer(samples, decrement)
            waveforms.append(waveform)
            samples -= len(waveform)

        waveform = np.concatenate(waveforms, axis=-1)
        return waveform, queue_empty


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
        super(InterleavedFIFOSignalQueue, self).__init__(*args, **kwargs)
        self._i = 0
        self._complete = False

    def decrement_key(self, key, n=1):
        super(InterleavedFIFOSignalQueue, self).decrement_key(key, n)
        if self._i >= len(self._ordering):
            self._i = 0

    def next_key(self):
        if self._complete:
            raise QueueEmptyError
        return self._ordering[self._i]

    def pop_next(self, decrement=True):
        key, data = super(InterleavedFIFOSignalQueue, self).pop_next(decrement)
        # Advance i only if the current key is not removed.  If the current key
        # was removed from _ordering, then the current value of i already
        # points to the next key in the sequence.
        self._i = (self._i + 1) % len(self._ordering)
        return key, data

    def decrement_key(self, key, n=1):
        if key not in self._ordering:
            raise KeyError('{} not in queue'.format(key))
        self._data[key]['trials'] -= n
        for data in self._data.values():
            if data['trials'] > 0:
                return
        self._complete = True


class RandomSignalQueue(AbstractSignalQueue):
    '''
    Return waveforms in random order
    '''

    def next_key(self):
        if len(self._ordering) == 0:
            raise QueueEmptyError
        i = np.random.randint(0, self.count_waveforms())
        return self._ordering[i]


queues = {
    'first-in, first-out': FIFOSignalQueue,
    'interleaved first-in, first-out': InterleavedFIFOSignalQueue,
    'random': RandomSignalQueue,
}
