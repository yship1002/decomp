"""
Features
    - all value functions are concave functions of y[0]

Problem formulation
    S in R^3, y in R^1, x in R^{3 x 1}
    min  sum_{s in S} a[s] * (y[0] - b[s]) ** 2

Solution
    y[0] = 10.33
    x[s, 0] = N/A (no impact), s in S
    obj = 8.36
"""

from ...main import StoModelBuilder


def const_model():

    scenarios = ['s1', 's2', 's3']
    y_set = [0]
    x_set = [0]
    Y = {
        0: [None, None],
    }

    X = {(s, 0): [None, None] for s in scenarios}

    a = {
        's1': 1,
        's2': 1,
        's3': 1,
    }

    b = {
        's1': 8,
        's2': 11,
        's3': 12
    }

    def obj1(m, s):
        return a[s] * (m.y[0] - b[s]) ** 2

    objs = {
        's1': obj1,
        's2': obj1,
        's3': obj1,
    }

    builder = StoModelBuilder('direct', name='concave_v', m_type='NLP', hint=False)

    _content = {
        'y_set': y_set,
        'x_set': x_set,
        'scenarios': scenarios,
        'y_bound': Y,
        'x_bound': X,
        'objs': objs,
        'con_2': {}
    }
    sto_m = builder.build(**_content)

    return sto_m
