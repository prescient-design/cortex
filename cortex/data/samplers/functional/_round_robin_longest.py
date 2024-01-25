import itertools
from abc import ABCMeta
from typing import Iterable, List, Sized


class SizedIterable(Sized, Iterable, metaclass=ABCMeta):
    pass


def round_robin_longest(iterables: List[SizedIterable]) -> Iterable:
    """Round robin of iterables until the longest have been exhausted.

    Example
    -------
    >>> iterables = [range(5), "ABCDE", ["cat", "dog", "rabbit"]]
    >>> iterator = round_robin_longest(iterables)
    >>> list(iterator)
    [0, 'A', 'cat', 1, 'B', 'dog', 2, 'C', 'rabbit', 3, 'D', 'cat', 4, 'E', 'dog']

    Parameters
    ----------
    iterables: list[SizedIterable]
        The iterables to roundly robin

    Returns
    -------
    Iterable
        The iterator stepping round robinly through.
        Cycles through shorter iterators.

    """
    max_len = max(len(iterable) for iterable in iterables)
    iterator_cycle = itertools.cycle(
        [
            itertools.cycle(iterable) if len(iterable) < max_len else iter(iterable)
            for iterable in iterables
            if len(iterable)
        ]
    )
    for iterator in iterator_cycle:
        try:
            yield next(iterator)
        except StopIteration:
            return
