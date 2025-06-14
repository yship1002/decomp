from itertools import product
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, Binary, NonNegativeReals, Set, Param
from ...main import StoModelBuilder

def const_model():

    pm = ConcreteModel('pure integer x')

    # generate scenarios
    omega_1_set = list(range(5, 15 + 1))
    omega_2_set = list(range(5, 15 + 1))
    total_omega_set = list(product(omega_1_set, omega_2_set))
    pm.S = Set(initialize=list(range(len(total_omega_set))))
    pm.prob = Param(pm.S, initialize=[1 / len(pm.S) for _ in pm.S])
    pm.omega_1 = Param(pm.S, initialize=[o[0] for o in total_omega_set])
    pm.omega_2 = Param(pm.S, initialize=[o[1] for o in total_omega_set])

    # first-stage variables
    pm.y1 = Var(within=NonNegativeReals, bounds=(0, 5))
    pm.y2 = Var(within=NonNegativeReals, bounds=(0, 5))

    # second-stage variables
    pm.x1 = Var(pm.S, within=Binary)
    pm.x2 = Var(pm.S, within=Binary)
    pm.x3 = Var(pm.S, within=Binary)
    pm.x4 = Var(pm.S, within=Binary)

    # constraints
    def con1(m, s):
        return 2 * m.x1[s] + 3 * m.x2[s] + 4 * m.x3[s] + 5 * m.x4[s] <= m.omega_1[s] - 1 / 3 * m.y1 - 2 / 3 * m.y2
    pm.con1 = Constraint(pm.S, rule=con1)
    def con2(m, s):
        return 6 * m.x1[s] + m.x2[s] + 3 * m.x3[s] + 2 * m.x4[s] <= m.omega_2[s] - 2 / 3 * m.y1 - 1 / 3 * m.y2
    pm.con2 = Constraint(pm.S, rule=con2)

    # objective
    def obj(m):
        return - 0.25 * 1.5 * m.y1 ** 2 - 0.25 * m.y2 ** 2 + sum(
            m.prob[s] * (
                - 16 * m.x1[s] - 19 * m.x2[s] - 23 * m.x3[s] - 28 * m.x4[s]
            )
        for s in m.S)
    pm.obj = Objective(rule=obj)

    # model transformation -----------------------------------------------------

    builder = StoModelBuilder('pyomo', name='pure_binary_x_nonlinear', m_type='MINLP', hint=False)

    scenarios = list(pm.S)
    var1_names = ['y1', 'y2']
    con1_names = []

    def _obj(m, s):
        return m.prob[s] * (
            - 0.25 * 1.5 * m.y1 ** 2 - 0.25 * 4 * m.y2 ** 2 - 16 * m.x1[s] - 19 * m.x2[s] - 23 * m.x3[s] - 28 * m.x4[s]
        )

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
