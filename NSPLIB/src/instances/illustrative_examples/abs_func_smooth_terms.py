"""
Features
    - value functions are absolute functions of y[0]
    - formulation only contains smooth terms
    - both CZ and LG are 1st order Hausdorff

Problem formulation
    S in R^2, y in R^1, x in R^{2 x 1}
    min (x[0, 0] * y[0] + 1) / 2 + x[1, 0] - 1
    s.t. x[1, 0] >= -y[0]
         x[1, 0] >= y[0]
         y[0], x[0, 0], x[1, 0] in [-1, 1]

Solution
    y[0] = 0
    x[0, 0] = 1
    x[1, 0] = 0
    obj = - 0.5
"""

from ...main import StoModelBuilder


def const_model():

    scenarios = ['s1', 's2']
    y_set = [0]
    x_set = [0]

    Y = {
        0: [-1, 1],
    }
    X = {(s, 0): [-1, 1] for s in scenarios}

    def con1(m, s):
        return m.x[s, 0] >= - m.y[0]

    def con2(m, s):
        return m.x[s, 0] >= m.y[0]

    def obj1(m, s):
        return (m.x[s, 0] * m.y[0] + 1) / 2

    def obj2(m, s):
        return ((m.x[s, 0] - 1) * 2) / 2

    objs = {
        's1': obj1,
        's2': obj2,
    }

    g_s = {
        's2': [con1, con2]
    }

    builder = StoModelBuilder('direct', name='abs_func_smooth', m_type='NLP', hint=False)

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
