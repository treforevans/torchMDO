from typing import Tuple


def is_broadcastable(shp1: Tuple[int], shp2: Tuple[int]) -> bool:
    """
    Determine if two array/tensor shapes can be broadcast from one to another.
    See: https://stackoverflow.com/a/24769712
    """
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True
