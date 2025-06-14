"""
Features
    - full problem is NLP
    - 2D y

Problem formulation
    S in R^3, y in R^2, x in R^{3 x 1}
    min  2.5 * (y[0] + y[1]) - sum_{s in S} d[s] * x[s, 0]
    s.t. c[s] / 2 * (y[0] + y[1]) - x[s, 0] >= 10           s in S
         (y[0] + y[1]) * x[s, 0] <= 40                      s in S
         c[s] * sqrt(x[s, 0] + 5) * log(x[s, 0] + 2) <= 35  s in S
         y[0], y[1] in [0, 20], x[0, 0], x[1, 0], x[2, 0] in [0, inf)

Solution
    y[0] = 13.06
    y[1] = 0
    x[0, 0] = 3.06
    x[1, 0] = 3.06
    x[2, 0] = 1.81
    obj = -113.81
"""

from pyomo.environ import sqrt, log
from ...main import StoModelBuilder


def const_model():
    scenarios = ['s1', 's2', 's3']
    y_set = [0, 1]
    x_set = [0]
    d = {
        's1': 2,
        's2': 4.5,
        's3': 10
    }
    c = {
        's1': 10,
        's2': 20,
        's3': 30
    }
    Y = {
        0: [0, 20],
        1: [0, 20]
    }

    X = {(s, 0): [0, None] for s in scenarios}

    def con1(m, s):
        return m.y[0] * d[s] / 2 + m.y[1] * d[s] / 2 - m.x[s, 0] >= 10

    def con2(m, s):
        return (m.y[0] + m.y[1]) * m.x[s, 0] <= 40

    def con3(m, s):
        return d[s] * sqrt(m.x[s, 0] + 5) * log(m.x[s, 0] + 2) <= 35

    def obj1(m, s):
        return 2.5 / 3 * m.y[0] + 2.5 / 3 * m.y[1] - c[s] * m.x[s, 0]

    objs = {
        's1': obj1,
        's2': obj1,
        's3': obj1
    }

    g_s = {
        's1': [con1, con2, con3],
        's2': [con1, con2, con3],
        's3': [con1, con2, con3]
    }

    builder = StoModelBuilder('direct', name='nonlinear-2D', m_type='NLP', hint=False)

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
