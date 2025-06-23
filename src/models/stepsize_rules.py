"""
This module provides the stepsize updating rules for the subgradient method.

Currently only the colorTV rule is implemented.
"""

from abc import ABC, abstractmethod
from typing import List
from src.utility.types import Multiplier

class StepsizeRule(ABC):
    """The general class of stepsize rules for subgradient methods.
    """

    @abstractmethod
    def update_stepsize_factor(self):
        pass


class ColorTVRule(StepsizeRule):
    """The class specifying the ColorTV stepsize rule for subgradient methods.

    Reference: https://link.springer.com/article/10.1007/s12532-017-0120-7

    Attributes:
        c_g (int): The threshold for consecutive green iteration number.
        c_y (int): The threshold for consecutive yellow iteration number.
        c_r (int): The threshold for consecutive red iteration number.
        factor_0 (float): The initial step size factor.
        factor_min (float): The minimum step size factor.
        factors (List[float]): The list of step sizes factor.
        n_g (int): The consecutive green iteration number.
        n_y (int): The consecutive yellow iteration number.
        n_r (int): The consecutive red iteration number.
        tol (float): A tolerance parameter.
    """

    def __init__(self):

        # self.factor_0 = 0.001
        self.factor_0 = 1.5
        self.factors = [self.factor_0, ]

        self.tol = 1e-6

        self.c_g = 10
        self.c_y = 50
        self.c_r = 10

        self.n_g = 0
        self.n_y = 0
        self.n_r = 0

        self.factor_min = 5e-8

    def _reset(self):
        self.factors = [self.factor_0, ]

    def update_stepsize_factor(self, direction: Multiplier, subgradient: Multiplier, lbds: List[float], best_lbd: float):
        """Update the stepsize factor.

        NOTE the factor here correspond to beta in the article, instead of
        nu (the final stepsize).

        Args:
            direction (Multiplier): The direction vector (a linear combination
            of current subgradient and last direction).
            subgradient (Multiplier): The subgradient from the Lagrangean
            subproblems.
            lbds (List[float]): All the lower bounds.
            best_lbd (float): The best lower bound.

        Returns:
            float: The updated stepsize factor.
        """

        # get scenarios and y set
        scenarios = list(direction.keys())
        y_set = list(direction[scenarios[0]].keys())

        # calculate the improvement
        try:
            improve = lbds[-1] - lbds[-2]
        except:
            improve = abs(lbds[-1])

        # calculate the product of direction and subgradient
        # it is used to estimate "how successful a step is"
        product = 0
        for s in scenarios:
            for y_idx in y_set:
                product += direction[s][y_idx] * subgradient[s][y_idx]

        # update color and n's
        improve_threshold = self.tol * max(best_lbd, 1)
        # print("green flag",self.n_g,"yellow flag",self.n_y,"red flag",self.n_r)
        # print("product",product,"improve",improve,"improve_threshold",improve_threshold)
        if (product > self.tol) and (improve > improve_threshold):
            self.n_y = self.n_r = 0
            self.n_g += 1
        elif (product < self.tol) and (improve >= 0):
            self.n_g = self.n_r = 0
            self.n_y += 1
        else:
            self.n_g = self.n_y = 0
            self.n_r += 1

        # update stepsize
        if self.n_g >= self.c_g:
            # factor = min(2.0, 2 * self.factors[-1])
            factor = 2 * self.factors[-1]

        elif self.n_y >= self.c_y:
            # factor = min(2.0, 1.1 * self.factors[-1])
            factor = 1.1 * self.factors[-1]

        elif self.n_r >= self.c_r:
            factor = max(self.factor_min, 0.67 * self.factors[-1])
        else:
            factor = self.factors[-1]

        # record new stepsize
        self.factors.append(factor)

        return factor


class PolyakRule(StepsizeRule):
    ...


class FumeroTVRule(StepsizeRule):
    ...
