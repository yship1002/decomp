"""
Features
    - value functions are non-Lipschitz

Problem formulation
    S in R^2, y in R^1, x in R^{2 x 1}
    min  x[1, 0] - x[0, 0]
    s.t. x[1, 0]^2 >= -y[0]
         x[1, 0]^2 >= y[0]
         x[0, 0]^4 = y[0]^2
         y[0] in [-1, 1], x[0, 0], x[1, 0] in [0, 1]

Solution
    y[0] = 0
    x[0, 0] = 0
    x[1, 0] = 0
    obj = 0

NOTE BARON cannot directly solve this problem, converged at x[1, 0] = 0.0255;
     seemed to be caused by the formulation (caused by rounding error);
     why BARON does not converge at (0, 0, 0), even when the initial values are
     set to that point?
"""

from ...main import StoModelBuilder


def const_model():
    scenarios = ['s1', 's2']
    y_set = [0]
    x_set = [0]
    Y = {
        0: [-1, 1],
    }

    X = {(s, 0): [0, 1] for s in scenarios}

    def con1(m, s):
        return m.x[s, 0] ** 2 >= - m.y[0]

    def con2(m, s):
        return m.x[s, 0] ** 2 >= m.y[0]

    def con3(m, s):
        return m.x[s, 0] ** 4 == m.y[0] ** 2

    def obj1(m, s):
        return - m.x[s, 0]

    def obj2(m, s):
        return m.x[s, 0]

    objs = {
        's1': obj1,
        's2': obj2,
    }

    g_s = {
        's1': [con3],
        's2': [con1, con2],
    }

    builder = StoModelBuilder('direct', name='non-lipschitz', m_type='NLP', hint=False)

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