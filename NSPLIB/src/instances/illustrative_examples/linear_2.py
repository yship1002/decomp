"""
Features
    - full problem is linear

Problem formulation
    S in R^2, y in R^1, x in R^{2 x 1}
    min  5 * y[0] - sum_{s in S} d[s] * x[s, 0]
    s.t. c[s] * y[0] - x[s, 0] >= 10    s in S
         x[s, 0] + y[0] <= 20           s in S
         y[0] in [0, 20]

Solution
    y[0] = 10
    x[0, 0] = 10
    x[1, 0] = 10
    obj = -100
"""

from ...main import StoModelBuilder


def const_model():

    c = {
        's1': 2,
        's2': 4.5
    }
    d = {
        's1': 10,
        's2': 5
    }

    scenarios = ['s1', 's2']

    y_set = [0]
    x_set = [0]

    Y = {
        0: [0, 20],
    }

    X = {(s, 0): [None, None] for s in scenarios}

    def con1(m, s):
        return m.y[0] * c[s] - m.x[s, 0] >= 10

    def con2(m, s):
        return m.y[0] + m.x[s, 0] <= 20

    def obj1(m, s):
        return 2.5 * m.y[0] - d[s] * m.x[s, 0]

    objs = {
        's1': obj1,
        's2': obj1,
    }

    g_s = {
        's1': [con1, con2],
        's2': [con1, con2],
    }

    builder = StoModelBuilder('direct', name='linear_2', m_type='MILP', hint=False)

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