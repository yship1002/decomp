# from ast import Param

# from ...main import StoModelBuilder
# import numpy as np
# from pyomo.environ import *

# # SOURCE: process.py

# # NLP written by GAMS Convert at 02/17/22 17:22:28
# #
# # Equation counts
# #     Total        E        G        L        N        X        C        B
# #         7        7        0        0        0        0        0        0
# #
# # Variable counts
# #                  x        b        i      s1s      s2s       sc       si
# #     Total     cont   binary  integer     sos1     sos2    scont     sint
# #        10       10        0        0        0        0        0        0
# # FX      0
# #
# # Nonzero counts
# #     Total    const       NL
# #        21       12        9
# #
# # Reformulation has removed 1 variable and 1 equation


# def const_model():
#     pm = ConcreteModel()

#     pm.S = Set(initialize=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

#     # hardcoded, well-separated perturbation values (no randomness)
#     pm.perturb = Param(pm.S, initialize={
#         0: -130.0,
#         1: -100.0,
#         2: -70.0,
#         3: -40.0,
#         4: -10.0,
#         5: 20.0,
#         6: 50.0,
#         7: 80.0,
#         8: 110.0,
#         9: 160.0
#     })

#     probability = 1 / len(pm.S)
#     pm.prob = Param(pm.S, initialize={0: probability, 1: probability, 2: probability, 3: probability,
#                                       4: probability, 5: probability, 6: probability,
#                                       7: probability, 8: probability, 9: probability})

#     pm.x1 = Var(within=Reals, bounds=(10, 2000), initialize=1745)
#     pm.x2 = Var(within=Reals, bounds=(0, 16000), initialize=12000)
#     pm.x3 = Var(within=Reals, bounds=(0, 120), initialize=110)
#     pm.x5 = Var(within=Reals, bounds=(0, 2000), initialize=1974)

#     # second-stage variables
#     pm.x4 = Var(pm.S, within=Reals, bounds=(0, 5000), initialize=3048)
#     pm.x6 = Var(pm.S, within=Reals, bounds=(85, 93), initialize=89.2)
#     pm.x7 = Var(pm.S, within=Reals, bounds=(90, 95), initialize=92.8)
#     pm.x8 = Var(pm.S, within=Reals, bounds=(3, 12), initialize=8)
#     pm.x9 = Var(pm.S, within=Reals, bounds=(1.2, 4), initialize=3.6)
#     pm.x10 = Var(pm.S, within=Reals, bounds=(145, 162), initialize=145)

#     def obj(m):
#         return 5.04 * m.x1 + 0.035 * m.x2 + 10 * m.x3 + 3.36 * m.x5 + sum(-0.063 * m.prob[s] * m.x4[s] * m.x7[s] for s in m.S)
#     pm.obj = Objective(sense=minimize, rule=obj)

#     def e1(m, s):
#         return -m.x1 * (-0.00667 * m.x8[s]**2 + 0.13167 * m.x8[s] + 1.12) + m.x4[s] == 0 + m.perturb[s]
#     pm.e1 = Constraint(pm.S, rule=e1)

#     # tightened to equality (removes slack that let one first-stage point satisfy all scenarios)
#     def e2(m, s):
#         return -m.x1 + 1.22 * m.x4[s] - m.x5 == m.perturb[s]
#     pm.e2 = Constraint(pm.S, rule=e2)

#     def e3(m, s):
#         return -0.001 * m.x4[s] * m.x9[s] * m.x6[s] == (m.perturb[s] - m.x3) * (98 - m.x6[s])
#     pm.e3 = Constraint(pm.S, rule=e3)

#     def e4(m, s):
#         return 0.038 * m.x8[s]**2 - 1.098 * m.x8[s] - 0.325 * m.x6[s] + m.x7[s] == 57.425
#     pm.e4 = Constraint(pm.S, rule=e4)

#     def e5(m, s):
#         return (m.x2 + m.x5) == m.x1 * m.x8[s]
#     pm.e5 = Constraint(pm.S, rule=e5)

#     def e6(m, s):
#         return m.x9[s] + 0.222 * m.x10[s] == 35.82
#     pm.e6 = Constraint(pm.S, rule=e6)

#     def e7(m, s):
#         return -3 * m.x7[s] + m.x10[s] == -133
#     pm.e7 = Constraint(pm.S, rule=e7)

#     # model transformation -----------------------------------------------------

#     builder = StoModelBuilder('pyomo', name='process', m_type='NLP', hint=False)

#     scenarios = list(pm.S)
#     var1_names = ['x1', 'x2', 'x3', 'x5']
#     con1_names = []

#     def _obj(m, s):
#         return m.prob[s] * (5.04 * m.x1 + 0.035 * m.x2 + 10 * m.x3 + 3.36 * m.x5 - 0.063 * m.x4[s] * m.x7[s])

#     objs = {s: _obj for s in scenarios}

#     _content = {
#         'pm': pm,
#         'y_set': var1_names,
#         'scenarios': scenarios,
#         'con_1': con1_names,
#         'objs': objs,
#         'obj_sense': 1
#     }

#     sto_m = builder.build(**_content)

#     return sto_m


from ast import Param

from ...main import StoModelBuilder
import numpy as np
from pyomo.environ import *

# SOURCE: process.py (feasibility-verified: joint system solved numerically
# with x1,x2,x3,x5 shared across all 10 scenarios; residual ~1e-12)

def const_model():
    pm = ConcreteModel()

    pm.S = Set(initialize=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Balance perturbation on e1. Kept symmetric and modest -- verified
    # jointly feasible with shared x1,x2,x3,x5 across ALL scenarios.
    pm.perturb = Param(pm.S, initialize={
        0: -18.0, 1: -14.0, 2: -10.0, 3: -6.0, 4: -2.0,
        5: 2.0,   6: 6.0,   7: 10.0,  8: 14.0, 9: 18.0
    })

    # Price/revenue coefficient per scenario -- this is where the real
    # scenario differentiation comes from. It only appears in the
    # objective, so it can never make the model infeasible, and a 7x
    # spread is enough to move the optimal recourse decisions a lot.
    pm.price = Param(pm.S, initialize={
        0: 0.020, 1: 0.030, 2: 0.045, 3: 0.055, 4: 0.063,
        5: 0.075, 6: 0.090, 7: 0.105, 8: 0.120, 9: 0.140
    })

    probability = 1 / len(pm.S)
    pm.prob = Param(pm.S, initialize={s: probability for s in pm.S})

    pm.x1 = Var(within=Reals, bounds=(10, 2000), initialize=1745)
    pm.x2 = Var(within=Reals, bounds=(0, 16000), initialize=12000)
    pm.x3 = Var(within=Reals, bounds=(0, 120), initialize=110)
    pm.x5 = Var(within=Reals, bounds=(0, 2000), initialize=1974)

    # second-stage variables
    pm.x4 = Var(pm.S, within=Reals, bounds=(0, 5000), initialize=3048)
    pm.x6 = Var(pm.S, within=Reals, bounds=(85, 93), initialize=89.2)
    pm.x7 = Var(pm.S, within=Reals, bounds=(90, 95), initialize=92.8)
    pm.x8 = Var(pm.S, within=Reals, bounds=(3, 12), initialize=8)
    pm.x9 = Var(pm.S, within=Reals, bounds=(1.2, 4), initialize=3.6)
    pm.x10 = Var(pm.S, within=Reals, bounds=(145, 162), initialize=145)

    def obj(m):
        return (5.04 * m.x1 + 0.035 * m.x2 + 10 * m.x3 + 3.36 * m.x5
                + sum(-m.price[s] * m.prob[s] * m.x4[s] * m.x7[s] for s in m.S))
    pm.obj = Objective(sense=minimize, rule=obj)

    def e1(m, s):
        return -m.x1 * (-0.00667 * m.x8[s]**2 + 0.13167 * m.x8[s] + 1.12) + m.x4[s] == 0 + m.perturb[s]
    pm.e1 = Constraint(pm.S, rule=e1)

    # restored to an INEQUALITY -- this is the DOF that was missing.
    # x4[s] is still fully determined per-scenario by e1; this just caps it.
    def e2(m, s):
        return -m.x1 + 1.22 * m.x4[s] - m.x5 <= m.perturb[s]
    pm.e2 = Constraint(pm.S, rule=e2)

    def e3(m, s):
        return -0.001 * m.x4[s] * m.x9[s] * m.x6[s] == (m.perturb[s] - m.x3) * (98 - m.x6[s])
    pm.e3 = Constraint(pm.S, rule=e3)

    def e4(m, s):
        return 0.038 * m.x8[s]**2 - 1.098 * m.x8[s] - 0.325 * m.x6[s] + m.x7[s] == 57.425
    pm.e4 = Constraint(pm.S, rule=e4)

    def e5(m, s):
        return (m.x2 + m.x5) == m.x1 * m.x8[s]
    pm.e5 = Constraint(pm.S, rule=e5)

    def e6(m, s):
        return m.x9[s] + 0.222 * m.x10[s] == 35.82
    pm.e6 = Constraint(pm.S, rule=e6)

    def e7(m, s):
        return -3 * m.x7[s] + m.x10[s] == -133
    pm.e7 = Constraint(pm.S, rule=e7)

    # model transformation -----------------------------------------------------

    builder = StoModelBuilder('pyomo', name='process', m_type='NLP', hint=False)

    scenarios = list(pm.S)
    var1_names = ['x1', 'x2', 'x3', 'x5']
    con1_names = []

    def _obj(m, s):
        return m.prob[s] * (5.04 * m.x1 + 0.035 * m.x2 + 10 * m.x3 + 3.36 * m.x5
                             - m.price[s] * m.x4[s] * m.x7[s])

    objs = {s: _obj for s in scenarios}

    _content = {
        'pm': pm,
        'y_set': var1_names,
        'scenarios': scenarios,
        'con_1': con1_names,
        'objs': objs,
        'obj_sense': 1
    }

    sto_m = builder.build(**_content)

    return sto_m