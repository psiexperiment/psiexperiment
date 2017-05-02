import logging
log = logging.getLogger(__name__)

from enaml.core.api import Declarative

import itertools
import copy
import uuid

import numpy as np


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

    def __init__(self, *args, **kwargs):
        self._data = {} # list of generators
        self._ordering = [] # order of items added to queue
        self._generator = None
        self._samples = 0
        self._delay = 0
        self._notifiers = []

    def _add_factory(self, factory, trials, delays, metadata):
        key = uuid.uuid4()
        data = {
            'factory': factory,
            'trials': trials,
            'delays': as_iterator(delays),
            'metadata': metadata,
        }
        self._data[key] = data
        return key

    def connect(self, callback):
        self._notifiers.append(callback)

    def insert(self, factory, trials, delays=None, metadata=None):
        k = self._add_factory(factory, trials, delays, metadata)
        self._ordering.insert(k)
        return k

    def append(self, factory, trials, delays=None, metadata=None):
        k = self._add_factory(factory, trials, delays, metadata)
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
            del self._data[key]
            self._ordering.remove(key)

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

        if samples > 0 and self._generator is not None:
                waveform, complete = self._generator.send({'samples': samples})
                samples -= len(waveform)
                self._samples += len(waveform)
                waveforms.append(waveform)
                if complete:
                    self._generator = None

        if samples > 0 and self._delay > 0:
                n_padding = min(self._delay, samples)
                waveform = np.zeros(n_padding)
                samples -= n_padding
                self._samples += len(waveform)
                self._delay -= n_padding
                waveforms.append(waveform)

        if (self._generator is None) and (self._delay == 0):
            try:
                key, data = self.pop_next(decrement=decrement)
                self._generator = data['factory']()
                next(self._generator)
                self._delay = next(data['delays'])
                for cb in self._notifiers:
                    args = self._samples, key, data['metadata']
                    cb(args)
            except QueueEmptyError:
                queue_empty = True

        if (samples > 0) and not queue_empty:
            waveform, queue_empty =  self.pop_buffer(samples, decrement)
            waveforms.append(waveform)

        waveform = np.concatenate(waveforms, axis=-1) 
        log.trace('Generated {} samples'.format(waveform.shape))
        if queue_empty:
            log.info('Queue is now empty')
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
        self.i = 0

    def decrement_key(self, key, n=1):
        super(InterleavedFIFOSignalQueue, self).decrement_key(key, n)
        if self.i >= len(self._ordering):
            self.i = 0

    def next_key(self):
        if len(self._ordering) == 0:
            raise QueueEmptyError
        return self._ordering[self.i]

    def pop_next(self, decrement=True):
        key, data = super(InterleavedFIFOSignalQueue, self).pop_next(decrement)
        # Advance i only if the current key is not removed.  If the current key
        # was removed from _ordering, then the current value of i already
        # points to the next key in the sequence.
        if data['trials'] != 0:
            self.i = (self.i + 1) % len(self._ordering)
        return key, data


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
    'FIFO': FIFOSignalQueue,
    'Interleaved FIFO': InterleavedFIFOSignalQueue,
    'Random': RandomSignalQueue,
}
