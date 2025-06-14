"""
Features
    - full problem is NLP
    - power functions as value functions

Problem formulation
    S in R^2, y in R^1, x in R^{2 x 1}
    min  2 * y_coef * y[0] ** y_power
    s.t. c[s] * y[0] - x[s, 0] >= 10                        s in S
         y[0] * x[s, 0] <= 20                               s in S
         c[s] * sqrt(x[s, 0] + 5) * log(x[s, 0] + 2) <= 35  s in S
         y[0] in [0, 20]

Solution
    y[0] = 5
    x[0, 0] = 0
    x[1, 0] = 0
    obj = 125
"""

from ...main import StoModelBuilder
from pyomo.environ import sqrt, log


def const_model(y_power=3):

    y_coef = 0.5

    c = {
        's1': 2,
        's2': 3
    }

    scenarios = ['s1', 's2']

    y_set = [0]
    x_set = [0]

    Y = {
        0: [0, 20],
    }

    X = {(s, 0): [0, None] for s in scenarios}

    def con1(m, s):
        return m.y[0] * c[s] - m.x[s, 0] >= 10

    def con2(m, s):
        return m.y[0] * m.x[s, 0] <= 20

    def con3(m, s):
        return c[s] * sqrt(m.x[s, 0] + 5) * log(m.x[s, 0] + 2) <= 35

    def obj1(m, s):
        return y_coef * m.y[0] ** y_power

    objs = {
        's1': obj1,
        's2': obj1,
    }

    g_s = {
        's1': [con1, con2, con3],
        's2': [con1, con2, con3],
    }

    builder = StoModelBuilder('direct', name='power_func_v', m_type='NLP', hint=False)

    _content = {
        'y_set': y_set,
        'x_set': x_set,
        'scenarios': scenarios,
        'con_2': g_s,
        'y_bound': Y,
        'x_bound': X,
        'objs': objs
    }
    sto_m = builder.build(**_content)

    return sto_m
