
from ...main import StoModelBuilder


# def const_model():
#     scenarios = ['s1', 's2']
#     y_set = [0]
#     x_set = [0]

#     Y = {
#         0: [0, 3]
#     }
#     c_3={
#         's1': 1.00694,
#         's2': -0.677232
#     }
#     c_2 = {
#         's1': -4.74589,
#         's2': 3.03949
#     }
#     c_1 = {
#         's1': 5.17523,
#         's2': -3.02338
#     }
#     X = {(s, 0): [0, 3] for s in scenarios}

#     def con1(m, s):
#         return m.y[0] - m.x[s,0] == 0



#     def obj1(m, s):
#         return c_3[s] * m.x[s,0]*m.x[s,0]*m.x[s,0] + c_2[s] * m.x[s, 0]*m.x[s, 0] + c_1[s] * m.x[s, 0]


#     objs = {
#         's1': obj1,
#         's2': obj1,
#     }

#     g_s = {
#         's1': [con1],
#         's2': [con1],
#     }

#     builder = StoModelBuilder('direct', name='fig12', m_type='NLP', hint=False)

#     _content = {
#         'y_set': y_set,
#         'x_set': x_set,
#         'scenarios': scenarios,
#         'con_2': g_s,
#         'y_bound': Y,
#         'x_bound': X,
#         'objs': objs
#     }
#     sto_m = builder.build(**_content)

#     return sto_m
import pyomo.environ as pyo
from functools import partial

# --- Top-level rule functions (picklable) ---
def con1_rule(m, s):
    return m.y[0] - m.x[s, 0] == 0

def obj_rule(m, s, c1, c2, c3):
    xs = m.x[s, 0]
    return c3[s] * xs**3 + c2[s] * xs**2 + c1[s] * xs


# --- Your builder function (no local rule defs) ---
def const_model():
    scenarios = ['s1', 's2']
    y_set = [0]
    x_set = [0]

    Y = {
        0: [0, 3]
    }
    c_3 = {
        's1': 1.00694,
        's2': -0.677232
    }
    c_2 = {
        's1': -4.74589,
        's2': 3.03949
    }
    c_1 = {
        's1': 5.17523,
        's2': -3.02338
    }
    X = {(s, 0): [0, 3] for s in scenarios}

    # Bind coefficient dicts once; reuse same callable per-scenario
    obj_with_coeffs = partial(obj_rule, c1=c_1, c2=c_2, c3=c_3)

    objs = {s: obj_with_coeffs for s in scenarios}
    g_s = {s: [con1_rule] for s in scenarios}

    builder = StoModelBuilder('direct', name='fig12', m_type='NLP', hint=False)

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
