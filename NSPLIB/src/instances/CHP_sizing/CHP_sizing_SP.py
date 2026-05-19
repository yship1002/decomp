import numpy as np
from ...main import StoModelBuilder
from pyomo.environ import ConcreteModel, Var, Constraint, Set, Objective, Param, Reals


def vertex_form(x, a, b, c):
    return c - b**2 / (4 * a) + a * (x + b / (2 * a))**2


# Problem constants
c_ref = 1.3e6
Qdot_nom_ref = 1.0
M = 0.9
c_m = 0.05
n = 30
i_rate = 0.05
T_OP = 6000
p_gas = 80
p_el_buy = 250
p_el_sell = 100
Qdot_nom_min = 1.4  # [MW], per paper Appendix A
Qdot_nom_max = 2.3  # [MW]
Qdot_rel_min = 0.5
Qdot_eps = 0.001 * Qdot_nom_max

af = ((1 + i_rate)**n * i_rate) / ((1 + i_rate)**n - 1)

# Scenario data: [Qdot_dem [MW], P_dem [MW]]
DEFAULT_DATA = [
    [1.1, 0.5], [1.1, 2.5], [1.1, 4.0],
    [1.2, 0.5], [1.2, 4.0],
    [1.3, 0.5], [1.3, 2.5], [1.3, 4.0],
]
Ns = len(DEFAULT_DATA)


def eff_th_nom(Qdot_nom):
    return 0.498 - Qdot_nom / 21.17

def eff_el_nom(Qdot_nom):
    return 0.372 + Qdot_nom / 21.17

def eff_th_rel(Qdot_rel):
    return vertex_form(Qdot_rel, -0.0768, -0.0199, 1.0960)

def eff_el_rel(Qdot_rel):
    return vertex_form(Qdot_rel, -0.2611, 0.6743, 0.5868)

def eff_th(Qdot_nom, Qdot_rel):
    return eff_th_nom(Qdot_nom) * eff_th_rel(Qdot_rel)

def eff_el(Qdot_nom, Qdot_rel):
    return eff_el_nom(Qdot_nom) * eff_el_rel(Qdot_rel)

def Edot_in_expr(Qdot_nom, Qdot_rel):
    return Qdot_nom * Qdot_rel / eff_th(Qdot_nom, Qdot_rel)

def P_out_expr(Qdot_nom, Qdot_rel):
    return Edot_in_expr(Qdot_nom, Qdot_rel) * eff_el(Qdot_nom, Qdot_rel)


# Bounds for auxiliary variables
P_out_max = P_out_expr(Qdot_nom_max, 1.0)
P_dem_max = max(d[1] for d in DEFAULT_DATA)


def const_model():
    pm = ConcreteModel()

    pm.S = Set(initialize=list(range(Ns)))
    pm.prob = Param(pm.S, initialize={s: 1.0 / Ns for s in range(Ns)})
    pm.Qdot_dem = Param(pm.S, initialize={s: DEFAULT_DATA[s][0] for s in range(Ns)})
    pm.P_dem = Param(pm.S, initialize={s: DEFAULT_DATA[s][1] for s in range(Ns)})

    # First-stage variable: nominal heat output [MW]
    pm.Qdot_nom = Var(within=Reals, bounds=(Qdot_nom_min, Qdot_nom_max), initialize=Qdot_nom_max)

    # Second-stage variable: relative heat output per scenario (reduced space)
    pm.Qdot_rel = Var(pm.S, within=Reals, bounds=(0.0, 1.0), initialize=1.0)

    # Auxiliary variables for power buy/sell (max() reformulation via power balance)
    pm.P_buy = Var(pm.S, within=Reals, bounds=(0.0, P_dem_max), initialize=0.0)
    pm.P_sell = Var(pm.S, within=Reals, bounds=(0.0, P_out_max), initialize=0.0)

    pm.obj = Objective(sense=1, expr=sum(
        pm.prob[s] * (
            1e-6 * (af + c_m) * c_ref * (pm.Qdot_nom / Qdot_nom_ref)**M
            + 1e-6 * T_OP * (
                p_gas * Edot_in_expr(pm.Qdot_nom, pm.Qdot_rel[s])
                + p_el_buy * pm.P_buy[s]
                - p_el_sell * pm.P_sell[s]
            )
        ) for s in pm.S
    ))

    def c_power_balance(m, s):
        return m.P_buy[s] - m.P_sell[s] - m.P_dem[s] + P_out_expr(m.Qdot_nom, m.Qdot_rel[s]) == 0
    pm.c_power_balance = Constraint(pm.S, rule=c_power_balance)

    def c_heat_demand(m, s):
        return m.Qdot_dem[s] - m.Qdot_nom * m.Qdot_rel[s] <= 0
    pm.c_heat_demand = Constraint(pm.S, rule=c_heat_demand)

    def c_min_partload(m, s):
        return vertex_form(m.Qdot_rel[s], -1.0, Qdot_rel_min + Qdot_eps, -Qdot_rel_min * Qdot_eps) <= 0
    pm.c_min_partload = Constraint(pm.S, rule=c_min_partload)

    # Model transformation -----------------------------------------------------

    builder = StoModelBuilder('pyomo', name='CHP_sizing', m_type='NLP', hint=False)

    scenarios = list(pm.S)
    var1_names = ['Qdot_nom']
    con1_names = []

    def _obj(m, s):
        return m.prob[s] * (
            1e-6 * (af + c_m) * c_ref * (m.Qdot_nom / Qdot_nom_ref)**M
            + 1e-6 * T_OP * (
                p_gas * Edot_in_expr(m.Qdot_nom, m.Qdot_rel[s])
                + p_el_buy * m.P_buy[s]
                - p_el_sell * m.P_sell[s]
            )
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
