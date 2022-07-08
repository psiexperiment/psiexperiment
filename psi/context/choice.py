'''
Introduction
------------
This module contains a collection of generators that facilitate ordering of
sequences for an experiment. Each generator takes a sequence and returns a
single element on each call. The order of the elements returned depends on the
algorithm. A shallow copy of the sequence is created by the generator to
prevent side effects if the original sequence is reordered elsewhere. However,
if the sequence contains mutable objects, then any modification to the objects
will be reflected in the output of the generator.

Examples
--------
Modifying a sequence of immutable objects after passing it to the selector does
not affect the output of the selector:

    >>> sequence = [1, 3, 8, 9, 12, 0, 4]
    >>> choice = exact_order(sequence)
    >>> next(choice)
    1
    >>> sequence[1] = 2
    >>> sequence
    [1, 2, 8, 9, 12, 0, 4]
    >>> next(choice)
    3

However, if you have a sequence that contains a mutable object (e.g. a list),
then modifications to the mutable object will alter the output of the selector:

    >>> sequence = [1, [2, 3], 4, 5, 6]
    >>> choice = exact_order(sequence)
    >>> next(choice)
    1
    >>> sequence[1][0] = -5
    >>> next(choice)
    [-5, 3]

An error is also raised when an empty sequence is passed:

    >>> choice = ascending([])
    Traceback (most recent call last):
        ...
    ValueError: Cannot use an empty sequence

Notes for developers
--------------------
* When writing your own generator, use the check_sequence decorator to ensure
  that the sequence passed is a shallow copy that you can modify in-place
  without any side effects. This also ensures that if the original sequence is
  modified by outside code, it will not affect the output of the generator.
* All generators must be infinite (i.e. they never end) or raise a StopIteration
  error when the sequence is exhausted.
* Random sequences have a hard dependency on Numpy (the built-in Python random
  module is suboptimal for scientific work).

'''
import logging
log = logging.getLogger(__name__)

from functools import wraps
import collections

import numpy as np
from numpy.random import RandomState


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

    Parameters
    ----------
    {common_docstring}

    Example
    -------
    >>> choice = ascending([1, 3, 8, 9, 12, 0, 4])
    >>> next(choice)
    0
    >>> next(choice)
    1
    >>> next(choice)
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

    Parameters
    ----------
    {common_docstring}

    Example
    -------
    >>> choice = descending([1, 3, 8, 9, 12, 0, 4])
    >>> next(choice)
    12
    >>> next(choice)
    9
    >>> next(choice)
    8
    '''
    sequence.sort(key=key, reverse=True)
    cycle = 0
    while cycle < c:
        for i in range(len(sequence)):
            yield sequence[i]
        cycle += 1


@check_sequence
def pseudorandom(sequence, c=np.inf, key=None, seed=None):
    '''
    Returns a randomly selected element from the sequence.

    Parameters
    ----------
    {common_docstring}
    seed : int
        Seed for random number generator.
    '''
    # We need to create a stand-alone generator that cannot be affected by
    # other parts of the code that may require random data (e.g. noise).
    state = RandomState()
    state.seed(seed)
    n = len(sequence)
    cycle = 0
    while cycle < c:
        i = state.randint(0, n)
        yield sequence[i]
        cycle += 1


@check_sequence
def exact_order(sequence, c=np.inf, key=None):
    '''
    Returns elements in the exact order they are provided.

    Parameters
    ----------
    {common_docstring}

    Example
    -------
    >>> choice = exact_order([1, 3, 8, 9, 12, 0, 4])
    >>> next(choice)
    1
    >>> next(choice)
    3
    >>> next(choice)
    8
    '''
    cycle = 0
    while cycle < c:
        for i in range(len(sequence)):
            yield sequence[i]
        cycle += 1


@check_sequence
def shuffled_set(sequence, c=np.inf, key=None, seed=None):
    '''
    Returns a randomly selected element from the sequence and removes it from
    the sequence.  Once the sequence is exhausted, repopulate list with the
    original sequence.

    Parameters
    ----------
    {common_docstring}
    seed : int
        Seed for random number generator.
    '''
    cycle = 0
    state = RandomState()
    state.seed(seed)
    while cycle < c:
        indices = list(range(len(sequence)))
        state.shuffle(indices)
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

    Parameters
    ----------
    {common_docstring}

    Example
    -------
    >>> import numpy as np
    >>> choice = counterbalanced([0, 1, 2], 60)
    >>> items = [next(choice) for i in range(60)]
    >>> np.bincount(items)
    array([20, 20, 20])

    >>> choice = counterbalanced(['RIGHT', 'LEFT'], 10)
    >>> items = [next(choice) for i in range(60)]
    >>> sorted(set(items))
    ['LEFT', 'RIGHT']
    >>> items.count('LEFT')
    30
    >>> items[:10].count('LEFT')
    5
    >>> items[10:20].count('LEFT')
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


common_docstring = '''
    sequence : {tuple, list}
        The iterable providing the sequence of values to be produced by the
        generator.
    c : {int, np.inf}
        Number of cycles to loop through sequence.
    key : {None, object}
        Sort key to use when determining ordering of generator. If None, default
        sort is used. This value is passed to the `key` parameter of `sort`.
'''

def format_docstrings():
    # Strip will remove the leading whitespace, thereby ensuring that the
    # common docstring remains properly indented since we already have leading
    # whitespace before {common_docstring} in the functions above.
    fmt = {
        'common_docstring': common_docstring.strip(),
    }
    for f in options.values():
        try:
            f.__doc__ = f.__doc__.format(**fmt)
        except:
            pass


format_docstrings()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
