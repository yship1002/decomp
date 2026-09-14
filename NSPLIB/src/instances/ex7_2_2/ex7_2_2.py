import numpy as np
from ...main import StoModelBuilder
from pyomo.environ import ConcreteModel, Var, Constraint, Set, Objective, Param, Reals



def const_model():
    np.random.seed(0)
    pm = ConcreteModel()

    pm.S = Set(initialize=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])


    pm.perturb = Param(pm.S, initialize={0: 10, 1: 20, 2: 30, 3: 40,
                                          4: 15, 5: 25, 6: 35, 7: 45, 8: 50, 
                                          9: 5, 10: 60, 11: 65, 12: 70, 13: 75,
                                         14: 80, 15: 90, 16: 95, 17: 55, 
                                         18: 100, 19: 105})
    probability = 1 / len(pm.S)
    pm.prob = Param(pm.S, initialize={0: probability, 1: probability, 2: probability, 3: probability, 4: probability, 5: probability, 6: probability,
                                       7: probability, 8: probability, 9: probability,10: probability, 11: probability, 12: probability, 13: probability,
                                         14: probability, 15: probability, 16: probability, 17: probability, 18: probability, 19: probability})


    # first-stage variables
    pm.x0 = Var( within=Reals, bounds=(0,1))
    pm.x1 = Var( within=Reals, bounds=(0,1))
    pm.x2 = Var(within=Reals, bounds=(0,1))
    pm.x3 = Var(within=Reals, bounds=(0,1))

    # second-stage variables
    pm.x4 = Var(pm.S,within=Reals, bounds=(1e-5, 16))
    pm.x5 = Var(pm.S,within=Reals, bounds=(1e-5, 16))


    pm.obj = Objective(sense=1, expr=sum(-1000000*pm.prob[s]*pm.x3 for s in pm.S))
    def e1(m,s):
        return 0.09755988*m.x1*m.x4[s]+m.x0-1== 0    #delete me this is just an experient to try different scnearios 
    pm.e1 = Constraint(pm.S, rule=e1)
    def e2(m,s):
        return 0.0965842812*m.x2*m.x5[s]+m.x1-m.x0==0
    pm.e2 = Constraint(pm.S, rule=e2)
    def e3(m,s):
        return 0.0391908*m.x2*m.x4[s]+m.x2+m.x0-1==0
    pm.e3 = Constraint(pm.S, rule=e3)
    def e4(m,s):
        return 0.03527172*m.x3*m.x5[s]+m.x3-m.x0+m.x1-m.x2==0
    pm.e4 = Constraint(pm.S, rule=e4)
    def e5(m,s):
        return m.x4[s]**0.5+m.x5[s]**0.5-m.perturb[s]<= 0
    pm.e5 = Constraint(pm.S, rule=e5)


    # model transformation -----------------------------------------------------

    builder = StoModelBuilder('pyomo', name='ex7_2_2', m_type='NLP', hint=False)

    scenarios = list(pm.S)

    var1_names = ['x' + str(i) for i in range(0,4)]
    con1_names = []

    def _obj(m, s):
        return -1000000*m.prob[s]*m.x3

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
# from ...main import StoModelBuilder
# from pyomo.environ import ConcreteModel, Var, Constraint, Set, Objective, Param, Reals


# def const_model():
#     pm = ConcreteModel()

#     n_scen = 5
#     pm.S = Set(initialize=list(range(n_scen)))

#     probability = 1 / n_scen
#     pm.prob = Param(pm.S, initialize={s: probability for s in range(n_scen)})

#     # ------------------------------------------------------------------
#     # Scenario data: explicit lists, one value per scenario, for each
#     # underlying uncertain quantity. Every scenario-dependent
#     # coefficient below is derived from these, so each scenario is a
#     # genuinely different instance of the physics/economics, not just
#     # one moving bound. Replace these lists with your own data
#     # (e.g. historical feed measurements) whenever you have it.
#     # ------------------------------------------------------------------

#     # "Feed"-type driver -> feeds e5's bound directly, and (via
#     # temp_factor below) the bilinear coefficients in e1-e4, since
#     # physically they'd respond to the same disturbance.
#     feed_level = [10, 20, 30, 40, 15]

#     # thermal-type driver, independent of feed_level, scales the
#     # bilinear coupling coefficients in e1-e4
#     temp_factor = [0.85, 0.90, 0.95, 1.00, 1.05]

#     # economic driver for the second-stage recourse cost
#     price_shock = [1.00, 1.05, 0.95, 1.10, 0.90]

#     perturb_vals = {s: float(feed_level[s]) for s in range(n_scen)}
#     pm.perturb = Param(pm.S, initialize=perturb_vals)

#     # Nominal coefficients, each perturbed by a scenario-specific factor
#     # (temperature-like driver, correlated but not identical per coefficient)
#     a1_vals = {s: 0.09755988 * temp_factor[s] for s in range(n_scen)}
#     a2_vals = {s: 0.0965842812 * temp_factor[s] ** 0.5 for s in range(n_scen)}
#     a3_vals = {s: 0.0391908 * temp_factor[s] for s in range(n_scen)}
#     a4_vals = {s: 0.03527172 * temp_factor[s] ** 0.5 for s in range(n_scen)}

#     pm.a1 = Param(pm.S, initialize=a1_vals)
#     pm.a2 = Param(pm.S, initialize=a2_vals)
#     pm.a3 = Param(pm.S, initialize=a3_vals)
#     pm.a4 = Param(pm.S, initialize=a4_vals)

#     # scenario-dependent recourse cost so x4, x5 actually trade off
#     # against something in the objective, not just satisfy equalities
#     recourse_cost_vals = {s: 1000 * price_shock[s] for s in range(n_scen)}
#     pm.recourse_cost = Param(pm.S, initialize=recourse_cost_vals)

#     # first-stage variables
#     pm.x0 = Var(within=Reals, bounds=(0, 1))
#     pm.x1 = Var(within=Reals, bounds=(0, 1))
#     pm.x2 = Var(within=Reals, bounds=(0, 1))
#     pm.x3 = Var(within=Reals, bounds=(0, 1))

#     # second-stage variables
#     pm.x4 = Var(pm.S, within=Reals, bounds=(1e-5, 16))
#     pm.x5 = Var(pm.S, within=Reals, bounds=(1e-5, 16))

#     pm.obj = Objective(
#         sense=1,
#         expr=sum(
#             pm.prob[s] * (-1000000 * pm.x3 + pm.recourse_cost[s] * (pm.x4[s] + pm.x5[s]))
#             for s in pm.S
#         ),
#     )

#     def e1(m, s):
#         return m.a1[s] * m.x1 * m.x4[s] + m.x0 - 1 == 0
#     pm.e1 = Constraint(pm.S, rule=e1)

#     def e2(m, s):
#         return m.a2[s] * m.x2 * m.x5[s] + m.x1 - m.x0 == 0
#     pm.e2 = Constraint(pm.S, rule=e2)

#     def e3(m, s):
#         return m.a3[s] * m.x2 * m.x4[s] + m.x2 + m.x0 - 1 == 0
#     pm.e3 = Constraint(pm.S, rule=e3)

#     def e4(m, s):
#         return m.a4[s] * m.x3 * m.x5[s] + m.x3 - m.x0 + m.x1 - m.x2 == 0
#     pm.e4 = Constraint(pm.S, rule=e4)

#     def e5(m, s):
#         return m.x4[s] ** 0.5 + m.x5[s] ** 0.5 - m.perturb[s] <= 0
#     pm.e5 = Constraint(pm.S, rule=e5)

#     # model transformation -----------------------------------------------------

#     builder = StoModelBuilder('pyomo', name='ex7_2_2', m_type='NLP', hint=False)

#     scenarios = list(pm.S)

#     var1_names = ['x' + str(i) for i in range(0, 4)]
#     con1_names = []

#     def _obj(m, s):
#         return m.prob[s] * (-1000000 * m.x3 + m.recourse_cost[s] * (m.x4[s] + m.x5[s]))

#     objs = {s: _obj for s in scenarios}

#     _content = {
#         'pm': pm,
#         'y_set': var1_names,
#         'scenarios': scenarios,
#         'con_1': con1_names,
#         'objs': objs,
#         'obj_sense': 1,
#     }

#     sto_m = builder.build(**_content)

#     return sto_m