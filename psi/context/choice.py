'''
Each generator takes a sequence and returns a single element on each call.  The
order of the elements returned depends on the algorithm.  The generators do not
modify the sequence.

* When adding a generator, use the check_sequence decorator to ensure
  that the sequence passed is a shallow copy that you can modify in-place
  without any side effects.
* All generators must be infinite (i.e. they never end) or raise a StopIteration
  error when the sequence is exhausted.
* Random sequences have a hard dependency on Numpy (the built-in Python random
  module is suboptimal for scientific work).
* If your sequence contains mutable objects, then any modifications to the
  objects themselves will be reflected in the output of the generator.

Examples
--------
Modifying a sequence of immutable objects after passing it to the selector does
not affect the output of the selector:

    >>> sequence = [1, 3, 8, 9, 12, 0, 4]
    >>> choice = exact_order(sequence)
    >>> choice.next()
    1
    >>> sequence[1] = 2
    >>> sequence
    [1, 2, 8, 9, 12, 0, 4]
    >>> choice.next()
    3

However, if you have a sequence that contains a mutable object (e.g. a list),
then modifications to the mutable object will alter the output of the selector:

    >>> sequence = [1, [2, 3], 4, 5, 6]
    >>> choice = exact_order(sequence)
    >>> print choice.next()
    1
    >>> sequence[1][0] = -5
    >>> print choice.next()
    [-5, 3]

An error is also raised when an empty sequence is passed:

    >>> choice = ascending([])
    Traceback (most recent call last):
        ...
    ValueError: Cannot use an empty sequence
'''

from functools import wraps
import doctest
import collections

import numpy as np


def check_sequence(f):
    '''
    Used to ensure that the sequence has at least one item and passes a shallow
    copy of the sequence to the selector so that we don't have side-effects if
    the sequence gets modified elsewhere in the program.
    '''
    @wraps(f)
    def wrapper(sequence, *args, **kw):
        if len(sequence) == 0:
            raise ValueError("Cannot use an empty sequence")
        sequence = sequence[:]
        return f(sequence, *args, **kw)
    return wrapper


@check_sequence
def ascending(sequence, c=np.inf, key=None):
    '''
    Returns elements from the sequence in ascending order.  When the last
    element is reached, loop around to the beginning.

    >>> choice = ascending([1, 3, 8, 9, 12, 0, 4])
    >>> choice.next()
    0
    >>> choice.next()
    1
    >>> choice.next()
    3
    '''
    sequence.sort(key=key)
    cycle = 0
    while cycle < c:
        for i in range(len(sequence)):
            yield sequence[i]
        cycle += 1


@check_sequence
def descending(sequence, c=np.inf, key=None):
    '''
    Returns elements from the sequence in descending order.  When the last
    element is reached, loop around to the beginning.

    >>> choice = descending([1, 3, 8, 9, 12, 0, 4])
    >>> choice.next()
    12
    >>> choice.next()
    9
    >>> choice.next()
    8
    '''
    sequence.sort(key=key, reverse=True)
    cycle = 0
    while cycle < c:
        for i in range(len(sequence)):
            yield sequence[i]
        cycle += 1


@check_sequence
def pseudorandom(sequence, seed=None, key=None):
    '''
    Returns a randomly selected element from the sequence.
    '''
    # We need to create a stand-alone generator that cannot be affected by other
    # parts of the code that may require random data (e.g. noise).
    from numpy.random import RandomState
    state = RandomState()
    state.seed(seed)
    n = len(sequence)
    while True:
        i = state.randint(0, n)
        yield sequence[i]


@check_sequence
def exact_order(sequence, c=np.inf, key=None):
    '''
    Returns elements in the exact order they are provided.

    >>> choice = exact_order([1, 3, 8, 9, 12, 0, 4])
    >>> choice.next()
    1
    >>> choice.next()
    3
    >>> choice.next()
    8
    '''
    cycle = 0
    while cycle < c:
        for i in range(len(sequence)):
            yield sequence[i]
        cycle += 1


@check_sequence
def shuffled_set(sequence, c=np.inf, key=None):
    '''
    Returns a randomly selected element from the sequence and removes it from
    the sequence.  Once the sequence is exhausted, repopulate list with the
    original sequence.
    '''
    cycle = 0
    while cycle < c:
        indices = range(len(sequence))
        np.random.shuffle(indices)  # Shuffle is in-place
        for i in indices:
            yield sequence[i]
        cycle += 1


@check_sequence
def counterbalanced(sequence, n, c=np.inf, key=None):
    '''
    Ensures that each value in `sequence` is presented an equal number of times
    over `n` trials.  At the end of the set, will regenerate a new list.  If you
    do not draw from this sequence a number of times that is a multiple of `n`,
    then the sequence will not be counterbalanced properly.

    >>> import numpy as np
    >>> choice = counterbalanced([0, 1, 2], 60)
    >>> print np.bincount([choice.next() for i in range(60)])
    [20 20 20]

    >>> choice = counterbalanced(['RIGHT', 'LEFT'], 10)
    >>> values = np.array([choice.next() for i in range(60)])
    >>> print np.unique(values)
    ['LEFT' 'RIGHT']
    >>> print len(values[values == 'LEFT'])
    30
    >>> print len(values[values[:10] == 'LEFT'])
    5

    '''
    sequence = np.asanyarray(sequence)
    cycle = 0
    while cycle < c:
        full_sequence = np.empty(n, dtype=sequence.dtype)
        sub_sequences = np.array_split(full_sequence, len(sequence))
        for s, value in zip(sub_sequences, sequence):
            s[:] = value
        np.random.shuffle(full_sequence)
        for s in full_sequence:
            yield s
        cycle += 1


options = collections.OrderedDict([
    ('ascending', ascending),
    ('descending', descending),
    ('pseudorandom', pseudorandom),
    ('exact_order', exact_order),
    ('counterbalanced', counterbalanced),
    ('shuffled_set', shuffled_set),
])


if __name__ == '__main__':
    doctest.testmod()
