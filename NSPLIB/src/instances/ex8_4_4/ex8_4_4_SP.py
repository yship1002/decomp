import numpy as np
from ...main import StoModelBuilder
from pyomo.environ import ConcreteModel, Var, Constraint, Set, Objective, Param, Reals



def const_model():
    np.random.seed(0)
    pm = ConcreteModel()

    pm.S = Set(initialize=[0, 1, 2])
    w = 0.1
    p1 = w * np.random.uniform(0, 1)
    p2 = w * np.random.uniform(1, 2)
    p3 = w * np.random.uniform(2, 3)
    pm.perturb = Param(pm.S, initialize={0: p1, 1: p2, 2: p3})
    pm.prob = Param(pm.S, initialize={0: 1 / 3, 1: 1 / 3, 2: 1 / 3})

    # first-stage variables
    pm.x6 = Var(within=Reals, bounds=(-2, 0), initialize=-1.551894266)
    pm.x7 = Var(within=Reals, bounds=(0.5, 2.5), initialize=1.199661008)
    pm.x8 = Var(within=Reals, bounds=(-1.5, 0.5), initialize=0.212540694)
    pm.x9 = Var(within=Reals, bounds=(0.2, 2.2), initialize=0.334227446)
    pm.x10 = Var(within=Reals, bounds=(-1.2, 0.8), initialize=-0.199578662)
    pm.x11 = Var(within=Reals, bounds=(0.1, 2.1), initialize=2.096235254)
    pm.x12 = Var(within=Reals, bounds=(-1.1, 0.9), initialize=0.057466756)
    pm.x13 = Var(within=Reals, bounds=(0, 1), initialize=0.991133039)
    pm.x14 = Var(within=Reals, bounds=(0, 1), initialize=0.762250467)
    pm.x15 = Var(within=Reals, bounds=(1.1, 1.3), initialize=1.1261384966)
    pm.x16 = Var(within=Reals, bounds=(0, 1), initialize=0.639718759)
    pm.x17 = Var(within=Reals, bounds=(0, 1), initialize=0.159517864)

    # second-stage variables
    pm.x1 = Var(pm.S, within=Reals, bounds=(4, 6), initialize=4.343494264)
    pm.x2 = Var(pm.S, within=Reals, bounds=(-6, -4), initialize=-4.313466584)
    pm.x3 = Var(pm.S, within=Reals, bounds=(2, 4), initialize=3.100750712)
    pm.x4 = Var(pm.S, within=Reals, bounds=(-3, -1), initialize=-2.397724192)
    pm.x5 = Var(pm.S, within=Reals, bounds=(1, 3), initialize=1.584424234)

    pm.obj = Objective(sense=1, expr=sum(pm.prob[s] * ((-5 + pm.x1[s])**2 + (5 + pm.x2[s])**2 + (-3 + pm.x3[s])**2 + (2 + pm.x4[s])**2 + (-2 + pm.x5[s])**2) for s in pm.S) +
                      (1 + pm.x6)**2 + (-1.5 + pm.x7) ** 2 + (0.5 + pm.x8)**2 + (-1.2 + pm.x9)**2 + (0.2 + pm.x10)**2 + (-1.1 + pm.x11) ** 2 + (0.1 + pm.x12)**2)

    pm.e4 = Constraint(expr=pm.x14 / 0.628318**pm.x15 - pm.x7 + pm.x13 == 0)
    pm.e5 = Constraint(expr=pm.x14 / 0.7853975**pm.x15 - pm.x9 + pm.x13 == 0)
    pm.e6 = Constraint(expr=pm.x14 / 0.942477**pm.x15 - pm.x11 + pm.x13 == 0)
    pm.e9 = Constraint(expr=-pm.x17 / 0.4712385**pm.x15 - pm.x6 + 0.4712385 * pm.x16
                      == 0)
    pm.e10 = Constraint(expr=-pm.x17 / 0.628318**pm.x15 - pm.x8 + 0.628318 * pm.x16
                       == 0)
    pm.e11 = Constraint(expr=-pm.x17 / 0.7853975**pm.x15 - pm.x10 + 0.7853975 * pm.x16
                       == 0)
    pm.e12 = Constraint(expr=-pm.x17 / 0.942477**pm.x15 - pm.x12 + 0.942477 * pm.x16
                       == 0)

    def e1(m, s):
        return m.x14 / 0.1570795**m.x15 - m.x1[s] + m.x13 == 0 + m.perturb[s]
    pm.e1 = Constraint(pm.S, rule=e1)

    def e2(m, s):
        return m.x14 / 0.314159**m.x15 - m.x3[s] + m.x13 == 0 + m.perturb[s]
    pm.e2 = Constraint(pm.S, rule=e2)

    def e3(m, s):
        return m.x14 / 0.4712385**m.x15 - m.x5[s] + m.x13 == 0 + m.perturb[s]
    pm.e3 = Constraint(pm.S, rule=e3)

    def e7(m, s):
        return -m.x17 / 0.1570795**m.x15 - m.x2[s] + 0.1570795 * m.x16 == 0
    pm.e7 = Constraint(pm.S, rule=e7)

    # def e8(m, s):
    #     return -m.x17 / 0.314159**m.x15 - m.x4[s] + 0.314159 * m.x16 == 0
    # m.e8 = Constraint(m.S, rule=e8)

    # model transformation -----------------------------------------------------

    builder = StoModelBuilder('pyomo', name='ex8_4_4', m_type='NLP', hint=False)

    scenarios = list(pm.S)

    var1_names = ['x' + str(i) for i in range(6, 17 + 1)]
    con1_names = ['e4', 'e5', 'e6', 'e9', 'e10', 'e11', 'e12']

    def _obj(m, s):
        return m.prob[s] * ((-5 + m.x1[s])**2 + (5 + m.x2[s])**2 + (-3 + m.x3[s])**2 + (2 + m.x4[s])**2 + (-2 + m.x5[s])**2 + (1 + m.x6)**2 + (-1.5 + m.x7) ** 2 + (0.5 + m.x8)**2 + (-1.2 + m.x9)**2 + (0.2 + m.x10)**2 + (-1.1 + m.x11) ** 2 + (0.1 + m.x12)**2)

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
