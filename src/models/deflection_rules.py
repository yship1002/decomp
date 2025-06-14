"""
This module provides classes for different deflection rules used in the
subgradient method.

Currently only the standard rule (alpha is always 1) is implemented.
"""
from abc import ABC, abstractmethod

class DeflectionRule(ABC):
    """The general class of deflection rules for subgradient methods.
    """

    @abstractmethod
    def deflect(self):
        pass


class STSubgradRule(DeflectionRule):
    """The class of the STSubgrad deflection rule for subgradient methods.
    """

    def __init__(self):
        self.alpha = 0.5  ### i changed it from 1 to 0.5

    def deflect(self):
        return self.alpha


class VolumeRule(DeflectionRule):
    ...


class PrimalDualRule(DeflectionRule):
    ...