"""
This module defines several common types used in the package for type hinting.
"""

from typing import Dict, Tuple, Union
from pyomo.environ import ConcreteModel

# bound for a single variable
Bound = Tuple[float, float]

# index for y
YIndex = Union[str, int]
# value of a single y point
YPoint = Dict[YIndex, float]
# bound for y (indexed)
YBound = Dict[YIndex, Bound]

# index for scenario
ScenarioIndex = Union[int, str]

# index for x
XIndex = Tuple[ScenarioIndex, Union[int, str]]

# pyomo model
PyomoModel = ConcreteModel

# multiplier for subgradient method
Multiplier = Dict[ScenarioIndex, Dict[YIndex, float]]
