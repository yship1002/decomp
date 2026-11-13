
import numpy as np
from ...main import StoModelBuilder
from pyomo.environ import ConcreteModel, Var, Constraint, Set, Objective, Param, Reals


def const_model():
    np.random.seed(0)
    pm = ConcreteModel()

    pm.S = Set(initialize=list(range(10)))

    # ------------------------------------------------------------------
    # CHANGE 1: independent perturbation vectors per constraint, instead
    # of one scalar reused in e1/e2/e3. Each is its own random draw, so
    # scenarios differ in *pattern*, not just magnitude of one number.
    # Also replaced the old {0: 1, 1: 2, 2: 0.03, ...} array (which had
    # two huge outliers next to eight tiny values) with a single
    # consistent, wide distribution.
    # ------------------------------------------------------------------



    perturb_a = {0: 0.16435402478574515, 1: -0.27625594348335564, 2: -0.5508317712765664, 3: -0.580166837365765, 4: 0.3759242870403269, 5: 0.49530669273326605, 6: 0.12796293092061584, 7: 0.27539587318079806, 8: 0.052349989758507444, 9: 0.5220869085453218}
    perturb_b = {0: 0.3790242649458386, 1: -0.5967137997958223, 2: 0.4288851319050832, 3: -0.5596973096334428, 4: 0.2755865357159329, 5: -0.3892132552769292, 6: 0.4358147068198639, 7: 0.049753464298910054, 8: -0.24034573135513826, 9: -0.09277533456280986}
    perturb_c = {0: -0.5660163946254444, 1: -0.45086006820052327, 2: 0.2047492976323564, 3: 0.1766274138891001, 4: 0.13846213377750463, 5: -0.13958693488573987, 6: 0.5966519229470532, 7: 0.577002406531476, 8: 0.22265038137683363, 9: 0.18055113152137955}
    perturb_d = {0: 0.22613607668512808, 1: -0.13329429122507544, 2: -0.4378841939731065, 3: 0.26578600823289805, 4: 0.030425186970871064, 5: -0.2277097493292532, 6: -0.016997569401853107, 7: 0.4673854012188003, 8: 0.5208522191474996, 9: -0.1706457639491157}
    perturb_e = {0: 0.08583579687571312, 1: -0.2137567307088694, 2: 0.11316003623963615, 3: -0.1945065293914401, 4: -0.13005719936620652, 5: 0.46832922240575076, 6: -0.3274108877599443, 7: 0.14782457362325088, 8: -0.4991815877011382, 9: 0.39917297718407735}


    pm.perturb_a = Param(pm.S, initialize=perturb_a)
    pm.perturb_b = Param(pm.S, initialize=perturb_b)
    pm.perturb_c = Param(pm.S, initialize=perturb_c)
    pm.perturb_d = Param(pm.S, initialize=perturb_d)
    pm.perturb_e = Param(pm.S, initialize=perturb_e)

    # ------------------------------------------------------------------
    # CHANGE 2: scenario-dependent objective targets. Previously every
    # scenario pulled x1..x5 toward the SAME fixed point (-5,5,-3,2,-2).
    # Now each scenario has its own target, so the second-stage cost
    # landscape itself differs across scenarios, not just the RHS of a
    # couple of constraints.
    # ------------------------------------------------------------------
    base_targets = {'x1': -5, 'x2': 5, 'x3': -3, 'x4': 2, 'x5': -2}

    target_shift={'x1': {0: 0.2739233746429086, 1: -0.4604265724722594, 2: -0.9180529521276106, 
            3: -0.9669447289429418, 4: 0.6265404784005448, 5: 0.8255111545554434, 
            6: 0.21327155153435973, 7: 0.4589931219679968, 8: 0.08724998293084574, 
            9: 0.8701448475755365},
    'x2': {0: 0.6317071082430643, 1: -0.9945229996597038, 2: 0.7148085531751387, 
           3: -0.9328288493890713, 4: 0.45931089285988813, 5: -0.648688758794882, 
           6: 0.7263578446997732, 7: 0.08292244049818343, 8: -0.40057621892523043,
             9: -0.1546255576046831}, 
    'x3': {0: -0.9433606577090741, 1: -0.7514334470008721, 2: 0.34124882938726064, 
           3: 0.2943790231485002, 4: 0.2307702229625077, 5: -0.23264489147623313, 
           6: 0.994419871578422, 7: 0.9616706775524602, 8: 0.3710839689613894, 
           9: 0.3009185525356326},
    'x4': {0: 0.37689346114188016, 1: -0.22215715204179243, 2: -0.7298069899551776, 
           3: 0.4429766803881634, 4: 0.050708644951451776, 5: -0.3795162488820887, 
           6: -0.028329282336421846, 7: 0.7789756686980005, 8: 0.8680870319124994,
             9: -0.28440960658185954},
    'x5': {0: 0.14305966145952187, 1: -0.3562612178481157, 2: 0.1886000603993936, 
           3: -0.3241775489857335, 4: -0.21676199894367754, 5: 0.7805487040095846, 
           6: -0.5456848129332406, 7: 0.24637428937208483, 8: -0.8319693128352303, 
           9: 0.6652882953067956}}
    pm.t1 = Param(pm.S, initialize={s: base_targets['x1'] + target_shift['x1'][s] for s in pm.S})
    pm.t2 = Param(pm.S, initialize={s: base_targets['x2'] + target_shift['x2'][s] for s in pm.S})
    pm.t3 = Param(pm.S, initialize={s: base_targets['x3'] + target_shift['x3'][s] for s in pm.S})
    pm.t4 = Param(pm.S, initialize={s: base_targets['x4'] + target_shift['x4'][s] for s in pm.S})
    pm.t5 = Param(pm.S, initialize={s: base_targets['x5'] + target_shift['x5'][s] for s in pm.S})

    # ------------------------------------------------------------------
    # CHANGE 3: non-uniform scenario probabilities (still sum to 1).
    # Uniform 0.1/0.1/... weights every scenario identically in the
    # master problem; unequal weights add another axis of distinction.
    # ------------------------------------------------------------------

    raw_w = [0.14337344, 0.08236041, 0.15333038, 0.06222044, 0.09313739, 0.07243643,
            0.10586093, 0.14440115, 0.08138826, 0.06149118]
    pm.prob = Param(pm.S, initialize={s: float(raw_w[s]) for s in pm.S})

    # first-stage variables (unchanged)
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

    # second-stage variables (unchanged)
    pm.x1 = Var(pm.S, within=Reals, bounds=(4, 6), initialize=4.343494264)
    pm.x2 = Var(pm.S, within=Reals, bounds=(-6, -4), initialize=-4.313466584)
    pm.x3 = Var(pm.S, within=Reals, bounds=(2, 4), initialize=3.100750712)
    pm.x4 = Var(pm.S, within=Reals, bounds=(-3, -1), initialize=-2.397724192)
    pm.x5 = Var(pm.S, within=Reals, bounds=(1, 3), initialize=1.584424234)

    # objective: now uses per-scenario targets t1..t5 instead of fixed constants
    pm.obj = Objective(
        sense=1,
        expr=sum(pm.prob[s] * ((pm.t1[s] + pm.x1[s])**2 + (pm.t2[s] + pm.x2[s])**2 +
                                (pm.t3[s] + pm.x3[s])**2 + (pm.t4[s] + pm.x4[s])**2 +
                                (pm.t5[s] + pm.x5[s])**2) for s in pm.S) +
             (1 + pm.x6)**2 + (-1.5 + pm.x7)**2 + (0.5 + pm.x8)**2 + (-1.2 + pm.x9)**2 +
             (0.2 + pm.x10)**2 + (-1.1 + pm.x11)**2 + (0.1 + pm.x12)**2
    )

    # first-stage-only constraints (unchanged)
    pm.e4 = Constraint(expr=pm.x14 / 0.628318**pm.x15 - pm.x7 + pm.x13 == 0)
    pm.e5 = Constraint(expr=pm.x14 / 0.7853975**pm.x15 - pm.x9 + pm.x13 == 0)
    pm.e6 = Constraint(expr=pm.x14 / 0.942477**pm.x15 - pm.x11 + pm.x13 == 0)
    pm.e9 = Constraint(expr=-pm.x17 / 0.4712385**pm.x15 - pm.x6 + 0.4712385 * pm.x16 == 0)
    pm.e10 = Constraint(expr=-pm.x17 / 0.628318**pm.x15 - pm.x8 + 0.628318 * pm.x16 == 0)
    pm.e11 = Constraint(expr=-pm.x17 / 0.7853975**pm.x15 - pm.x10 + 0.7853975 * pm.x16 == 0)
    pm.e12 = Constraint(expr=-pm.x17 / 0.942477**pm.x15 - pm.x12 + 0.942477 * pm.x16 == 0)

    # scenario constraints: each now uses its OWN perturbation param
    def e1(m, s):
        return m.x14 / 0.1570795**m.x15 - m.x1[s] + m.x13 == 0 + m.perturb_a[s]
    pm.e1 = Constraint(pm.S, rule=e1)

    def e2(m, s):
        return m.x14 / 0.314159**m.x15 - m.x3[s] + m.x13 == 0 + m.perturb_b[s]
    pm.e2 = Constraint(pm.S, rule=e2)

    def e3(m, s):
        return m.x14 / 0.4712385**m.x15 - m.x5[s] + m.x13 == 0 + m.perturb_c[s]
    pm.e3 = Constraint(pm.S, rule=e3)

    def e7(m, s):
        return -m.x17 / 0.1570795**m.x15 - m.x2[s] + 0.1570795 * m.x16 == 0 + m.perturb_d[s]
    pm.e7 = Constraint(pm.S, rule=e7)

    # CHANGE 4: e8 is back in and now actually constrains x4[s] per scenario.
    # Previously this was commented out, leaving x4 completely free (same
    # trivial unconstrained optimum in every scenario).
    def e8(m, s):
        return -m.x17 / 0.314159**m.x15 - m.x4[s] + 0.314159 * m.x16 == 0 + m.perturb_e[s]
    pm.e8 = Constraint(pm.S, rule=e8)

    # model transformation -----------------------------------------------------

    builder = StoModelBuilder('pyomo', name='ex8_4_4', m_type='NLP', hint=False)

    scenarios = list(pm.S)

    var1_names = ['x' + str(i) for i in range(6, 17 + 1)]
    con1_names = ['e4', 'e5', 'e6', 'e9', 'e10', 'e11', 'e12']

    def _obj(m, s):
        return m.prob[s] * ((m.t1[s] + m.x1[s])**2 + (m.t2[s] + m.x2[s])**2 +
                             (m.t3[s] + m.x3[s])**2 + (m.t4[s] + m.x4[s])**2 +
                             (m.t5[s] + m.x5[s])**2 +
                             (1 + m.x6)**2 + (-1.5 + m.x7)**2 + (0.5 + m.x8)**2 +
                             (-1.2 + m.x9)**2 + (0.2 + m.x10)**2 + (-1.1 + m.x11)**2 +
                             (0.1 + m.x12)**2)

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












# def const_model():
#     np.random.seed(0)
#     pm = ConcreteModel()

#     pm.S = Set(initialize=[0, 1, 2,3,4,5,6,7,8,9])
#     # pm.S = Set(initialize=[0, 1, 2])


#     pm.perturb = Param(pm.S, initialize={0: 1, 1: 2, 2: 0.03, 3: 0.04, 
#                     4: 0.05, 5: 0.06, 6: 0.07, 7: 0.08, 8: 0.09, 9: 0.1})

#     pm.prob = Param(pm.S, initialize={0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1})
#     # pm.perturb = Param(pm.S, initialize={0: 0.05488135039273248, 1: 0.17151893663724196, 
#     #         2: 0.2602763376071644})
#     # pm.prob = Param(pm.S, initialize={0: 1.0/3, 1: 1.0/3, 2: 1.0/3})

#     # first-stage variables
#     pm.x6 = Var(within=Reals, bounds=(-2, 0), initialize=-1.551894266)
#     pm.x7 = Var(within=Reals, bounds=(0.5, 2.5), initialize=1.199661008)
#     pm.x8 = Var(within=Reals, bounds=(-1.5, 0.5), initialize=0.212540694)
#     pm.x9 = Var(within=Reals, bounds=(0.2, 2.2), initialize=0.334227446)
#     pm.x10 = Var(within=Reals, bounds=(-1.2, 0.8), initialize=-0.199578662)
#     pm.x11 = Var(within=Reals, bounds=(0.1, 2.1), initialize=2.096235254)
#     pm.x12 = Var(within=Reals, bounds=(-1.1, 0.9), initialize=0.057466756)
#     pm.x13 = Var(within=Reals, bounds=(0, 1), initialize=0.991133039)
#     pm.x14 = Var(within=Reals, bounds=(0, 1), initialize=0.762250467)
#     pm.x15 = Var(within=Reals, bounds=(1.1, 1.3), initialize=1.1261384966)
#     pm.x16 = Var(within=Reals, bounds=(0, 1), initialize=0.639718759)
#     pm.x17 = Var(within=Reals, bounds=(0, 1), initialize=0.159517864)

#     # second-stage variables
#     pm.x1 = Var(pm.S, within=Reals, bounds=(4, 6), initialize=4.343494264)
#     pm.x2 = Var(pm.S, within=Reals, bounds=(-6, -4), initialize=-4.313466584)
#     pm.x3 = Var(pm.S, within=Reals, bounds=(2, 4), initialize=3.100750712)
#     pm.x4 = Var(pm.S, within=Reals, bounds=(-3, -1), initialize=-2.397724192)
#     pm.x5 = Var(pm.S, within=Reals, bounds=(1, 3), initialize=1.584424234)

#     pm.obj = Objective(sense=1, expr=sum(pm.prob[s] * ((-5 + pm.x1[s])**2 + (5 + pm.x2[s])**2 + (-3 + pm.x3[s])**2 + (2 + pm.x4[s])**2 + (-2 + pm.x5[s])**2) for s in pm.S) +
#                       (1 + pm.x6)**2 + (-1.5 + pm.x7) ** 2 + (0.5 + pm.x8)**2 + (-1.2 + pm.x9)**2 + (0.2 + pm.x10)**2 + (-1.1 + pm.x11) ** 2 + (0.1 + pm.x12)**2)

#     pm.e4 = Constraint(expr=pm.x14 / 0.628318**pm.x15 - pm.x7 + pm.x13 == 0)
#     pm.e5 = Constraint(expr=pm.x14 / 0.7853975**pm.x15 - pm.x9 + pm.x13 == 0)
#     pm.e6 = Constraint(expr=pm.x14 / 0.942477**pm.x15 - pm.x11 + pm.x13 == 0)
#     pm.e9 = Constraint(expr=-pm.x17 / 0.4712385**pm.x15 - pm.x6 + 0.4712385 * pm.x16
#                       == 0)
#     pm.e10 = Constraint(expr=-pm.x17 / 0.628318**pm.x15 - pm.x8 + 0.628318 * pm.x16
#                        == 0)
#     pm.e11 = Constraint(expr=-pm.x17 / 0.7853975**pm.x15 - pm.x10 + 0.7853975 * pm.x16
#                        == 0)
#     pm.e12 = Constraint(expr=-pm.x17 / 0.942477**pm.x15 - pm.x12 + 0.942477 * pm.x16
#                        == 0)

#     def e1(m, s):
#         return m.x14 / 0.1570795**m.x15 - m.x1[s] + m.x13 == 0 + m.perturb[s]
#     pm.e1 = Constraint(pm.S, rule=e1)

#     def e2(m, s):
#         return m.x14 / 0.314159**m.x15 - m.x3[s] + m.x13 == 0 + m.perturb[s]
#     pm.e2 = Constraint(pm.S, rule=e2)

#     def e3(m, s):
#         return m.x14 / 0.4712385**m.x15 - m.x5[s] + m.x13 == 0 + m.perturb[s]
#     pm.e3 = Constraint(pm.S, rule=e3)

#     def e7(m, s):
#         return -m.x17 / 0.1570795**m.x15 - m.x2[s] + 0.1570795 * m.x16 == 0
#     pm.e7 = Constraint(pm.S, rule=e7)

#     # def e8(m, s):
#     #     return -m.x17 / 0.314159**m.x15 - m.x4[s] + 0.314159 * m.x16 == 0
#     # m.e8 = Constraint(m.S, rule=e8)

#     # model transformation -----------------------------------------------------

#     builder = StoModelBuilder('pyomo', name='ex8_4_4', m_type='NLP', hint=False)

#     scenarios = list(pm.S)

#     var1_names = ['x' + str(i) for i in range(6, 17 + 1)]
#     con1_names = ['e4', 'e5', 'e6', 'e9', 'e10', 'e11', 'e12']

#     def _obj(m, s):
#         return m.prob[s] * ((-5 + m.x1[s])**2 + (5 + m.x2[s])**2 + (-3 + m.x3[s])**2 + (2 + m.x4[s])**2 + (-2 + m.x5[s])**2 + (1 + m.x6)**2 + (-1.5 + m.x7) ** 2 + (0.5 + m.x8)**2 + (-1.2 + m.x9)**2 + (0.2 + m.x10)**2 + (-1.1 + m.x11) ** 2 + (0.1 + m.x12)**2)

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
