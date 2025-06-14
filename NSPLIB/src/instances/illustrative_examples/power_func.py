"""
Features
    - power functions as value functions and are represented with x terms

Problem formulation
    S in R^2, y in R^1, x in R^{2 x 1}
    min  x[0, 0]&p - x[1, 0]^p
    s.t. x[s, 0] = y[0]         s in S
         y[0] in [0, 20]

Solution
    y[0] = 0
    x[0, 0] = 0
    x[1, 0] = 0
    obj = 0
"""

from ...main import StoModelBuilder


def const_model(x_power=2):

    scenarios = ['s1', 's2']

    y_set = [0]
    x_set = [0]

    Y = {
        0: [0, 20],
    }

    X = {(s, 0): [None, None] for s in scenarios}

    def con1(m, s):
        return m.x[s, 0] == m.y[0]

    def obj1(m, s):
        return m.x[s, 0] ** x_power

    def obj2(m, s):
        return - m.x[s, 0] ** x_power

    objs = {
        's1': obj1,
        's2': obj2,
    }

    g_s = {
        's1': [con1],
        's2': [con1],
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
