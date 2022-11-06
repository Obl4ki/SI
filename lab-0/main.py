from typing import Iterable
import numpy as np
from itertools import product


def is_unique(iterable: Iterable):
    return len(set(iterable)) == len(iterable)


def can_beat(queens: np.ndarray):
    if not is_unique(queens):
        return True

    for (idx1, queen1), (idx2, queen2) in product(enumerate(queens), enumerate(queens)):
        if queen1 == queen2:
            continue
        if abs(idx1 - idx2) == abs(queen1 - queen2):
            return True

    return False


if __name__ == '__main__':
    queens_cant_beat = np.array([4, 0, 7, 3, 1, 6, 2, 5])
    queens_can_beat = np.array([4, 2, 5, 5, 6, 7, 1])

    assert not can_beat(queens_cant_beat)
    assert can_beat(queens_can_beat)
