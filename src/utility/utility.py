"""
This module contains utility functions that are used in the project.
Currently two methods are defined: norm and filter_infty.
"""
from src.utility.types import Multiplier


def norm(y: Multiplier) -> float:
    """
    Calculate the l-2 norm of the given multiplier.
    args:
        y: The multiplier to be calculated.
    returns:
        float: The l-2 norm of the multiplier.
    """

    sum = 0
    scenarios = list(y.keys())
    y_indices = list(y[scenarios[0]].keys())

    for s in scenarios:
        for y_idx in y_indices:
            sum += y[s][y_idx] ** 2

    return sum ** 0.5

def filter_infty(n: float):
    """
    Return infinity if n is greater than 1e20. Sometimes barron returns 1e51 or
    2e51 for upper bound.
    args:
        n: The number to be filtered.
    returns:
        float: The filtered number. or infinity if n is greater than 1e20.
    """

    return n if n < 1e20 else float("inf")