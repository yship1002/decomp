"""
Features
    - value functions are absolute functions of y[0]
    - both CZ and LG are 1st order Hausdorff

Problem formulation
    S in R^2, y in R^1, x in R^{2 x 1}
    min  (x[0, 0] - 10)^2^0.5 - (x[1, 0] - 10)^2^0.5
    s.t. x[s, 0] = y[0]     s in S
         y[0], x[0, 0], x[1, 0] in [-10, 10]

Solution
    y[0] = 10
    x[0, 0] = 10
    x[1, 0] = 10
    obj = 0
"""

from ...main import StoModelBuilder
from pyomo.environ import sqrt


def const_model():
    scenarios = ['s1', 's2']
    y_set = [0]
    x_set = [0]
    Y = {
        0: [-10, 10],
    }

    X = {(s, 0): [-1, 1] for s in scenarios}

    def con1(m, s):
        return m.x[s, 0] == m.y[0]

    def obj1(m, s):
        return sqrt((m.x[s, 0] - 10) ** 2)

    def obj2(m, s):
        return - sqrt((m.x[s, 0] - 10) ** 2)

    objs = {
        's1': obj1,
        's2': obj2,
    }

    g_s = {
        's1': [con1],
        's2': [con1],
    }

    builder = StoModelBuilder('direct', name='abs_func', m_type='NLP', hint=False)

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