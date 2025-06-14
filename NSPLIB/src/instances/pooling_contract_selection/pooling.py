"""

correspondence of variables in GitHub (left) & paper (right)
    sets
        W ----------------- Omega (represented as `S` below)
        None -------------- C (contracts, represented as separated variable/constraints for f, b, d)
    first-stage variables
        gamma_intlt ------- lambda (represented as `lambd` below)
        gamma_pool -------- theta
    parameters
        CC ---------------- C_{i, k}
        c_fixed_inlt ------ delta_i
        c_fixed_pool ------ alpha_i
        c_variable_inlt --- xi_i
        c_variable_pool --- beta_i
        prob -------------- tau_s

source
    https://link.springer.com/article/10.1007/s10898-019-00816-8
    https://github.com/bbrunaud/PlasmoAlgorithms.jl/blob/clrootnode/examples/PoolingContract/fullspace3.gms
    Can Li, Ignacio Grossmann
"""
from pyomo.environ import RangeSet, ConcreteModel, Var, NonNegativeReals, Binary, Constraint, Objective
from ...main import StoModelBuilder


def const_model():
    """Construct the model. This model is first built as a Pyomo model and then
    transferred into StochasticModel.
    """

    pm = ConcreteModel()

    # sets ---------------------------------------------------------------------

    # feeds
    I = RangeSet(1, 5)
    # products
    J = RangeSet(1, 3)
    # pools
    L = RangeSet(1, 4)
    # qualities
    K = RangeSet(1, 2)
    # base scenarios; formulated as W in .gms
    S = RangeSet(1, 3)
    Tx = I * L
    Ty = L * J
    Tz = RangeSet() * RangeSet()

    # parameters ---------------------------------------------------------------

    # minimum available flow
    AL = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }
    # maximum available flow
    AU = {
        1: 300,
        2: 250,
        3: 0,
        4: 0,
        5: 300,
    }
    # minimum pool size
    SL = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
    # maximum pool size
    SU = {
        1: 400,
        2: 0,
        3: 0,
        4: 500,
    }
    # unit price of product j (base)
    base_d = {
        1: 5.7,
        2: 6.2,
        3: 6.8
    }
    # maximum demand for product j (base)
    base_DU = {
        1: 229,
        2: 173,
        3: 284
    }
    # fixed cost for pools
    alpha = {
        1: 310,
        2: 470,
        3: 380,
        4: 510
    }
    # variable cost for pools
    beta = {
        1: 1.1,
        2: 0.9,
        3: 1.05,
        4: 0.8,
    }
    # fixed cost for feed storage
    delta = {
        1: 260,
        2: 70,
        3: 150,
        4: 190,
        5: 110
    }
    # variable cost for feed storage
    xi = {
        1: 0.5,
        2: 0.8,
        3: 0.6,
        4: 0.55,
        5: 0.7
    }

    # helper function to flatten nested dict
    def flatten(nested_dict):

        new_dict = {}
        for i in nested_dict:
            for j in nested_dict[i]:
                new_dict[i, j] = nested_dict[i][j]

        return new_dict

    # feed concentration
    C = {
        1: {
            1: 0.13,
            2: 0.87,
        },
        2: {
            1: 0.89,
            2: 0.11,
        },
        3: {
            1: 0.69,
            2: 0.31,
        },
        4: {
            1: 0.28,
            2: 0.72,
        },
        5: {
            1: 0.35,
            2: 0.65,
        },
    }
    C = flatten(C)

    # maximum allowable product concentration
    PU = {
        1: {
            1: 0.56,
            2: 0.44,
        },
        2: {
            1: 0.30,
            2: 0.70,
        },
        3: {
            1: 0.41,
            2: 0.59,
        },
    }
    PU = flatten(PU)

    # minimum allowable product concentration
    PL = {
        1: {
            1: 0.56,
            2: 0.44,
        },
        2: {
            1: 0.30,
            2: 0.70,
        },
        3: {
            1: 0.41,
            2: 0.59,
        },
    }
    PL = flatten(PL)

    base_tau = {
        1: 0.3,
        2: 0.4,
        3: 0.3
    }

    base_psi_f = {
        1: 0.5,
        2: 0.5,
        3: 0.5,
        4: 0.5,
        5: 0.5,
    }
    base_psi_d1 = {
        1: 0.55,
        2: 0.55,
        3: 0.55,
        4: 0.55,
        5: 0.55,
    }
    base_psi_d2 = {
        1: 0.4,
        2: 0.4,
        3: 0.4,
        4: 0.4,
        5: 0.4,
    }
    base_psi_b1 = {
        1: 0.55,
        2: 0.55,
        3: 0.55,
        4: 0.55,
        5: 0.55,
    }
    base_psi_b2 = {
        1: 0.48,
        2: 0.48,
        3: 0.48,
        4: 0.48,
        5: 0.48,
    }
    ratio = {
        1: 0.7,
        2: 1.0,
        3: 1.3
    }
    # minimum purchasable amount of feed i under contract b/d
    sigma_b = {}
    sigma_d = {}
    for i in I:
        sigma_b[i] = AU[i] / 2
        sigma_d[i] = AU[i] / 3 * 2
    # final probability
    tau = base_tau.copy()

    # maximum product demand
    DU = {}
    # unit price of product j in scenario s
    d = {}
    for j in J:
        for s in S:
            DU[j, s] = base_DU[j] * ratio[s]
            d[j, s] = base_d[j]

    # unit price for feed i under contract f/d1/d2/b1/b2
    psi_f = {}
    psi_d1 = {}
    psi_d2 = {}
    psi_b1 = {}
    psi_b2 = {}
    for i in I:
        for s in S:
            psi_f[i, s] = base_psi_f[i]
            psi_d1[i, s] = base_psi_d1[i]
            psi_d2[i, s] = base_psi_d2[i]
            psi_b1[i, s] = base_psi_b1[i]
            psi_b2[i, s] = base_psi_b2[i]

    # variables ----------------------------------------------------------------
    # first-stage variables
    # capacity of pool l
    def _S_bound_rule(m, l):
        return (SL[l], SU[l])
    pm.S = Var(L, bounds=_S_bound_rule, within=NonNegativeReals)
    # capacity of pool i
    def _A_bound_rule(m, i):
        return (AL[i], AU[i])
    pm.A = Var(I, bounds=_A_bound_rule, within=NonNegativeReals)
    # whether feed i exists
    pm.lambd = Var(I, within=Binary)
    # whether pool l exists
    pm.theta = Var(L, within=Binary)

    # second-stage variables
    # flow from pool l to product j in scenario s
    pm.y = Var(L, J, S, within=NonNegativeReals)
    # flow from feed i to product j in scenario s
    pm.z = Var(I, J, S, within=NonNegativeReals)
    # proportion of flow from feed i to pool l in scenario s
    pm.q = Var(I, L, S, within=NonNegativeReals)
    # cost of purchasing feed i in scenario s
    pm.CT = Var(I, S, within=NonNegativeReals)
    # cost of purchasing feed i under contract f in scenario s
    pm.CTf = Var(I, S, within=NonNegativeReals)
    # cost of purchasing feed i under contract b in scenario s
    pm.CTb = Var(I, S, within=NonNegativeReals)
    # cost of purchasing feed i under contract d in scenario s
    pm.CTd = Var(I, S, within=NonNegativeReals)
    # the amount of feed i purchased under contract f in scenario s
    pm.Bf = Var(I, S, within=NonNegativeReals)
    # the amount of feed i purchased under contract d in scenario s
    pm.Bd = Var(I, S, within=NonNegativeReals)
    pm.Bd1 = Var(I, S, within=NonNegativeReals)
    pm.Bd2 = Var(I, S, within=NonNegativeReals)
    pm.Bd11 = Var(I, S, within=NonNegativeReals)
    pm.Bd12 = Var(I, S, within=NonNegativeReals)
    # the amount of feed i purchased under contract b in scenario s
    pm.Bb = Var(I, S, within=NonNegativeReals)
    pm.Bb1 = Var(I, S, within=NonNegativeReals)
    pm.Bb2 = Var(I, S, within=NonNegativeReals)
    # whether contract f is selected in purchasing feed i in scenario s
    pm.uf = Var(I, S, within=Binary)
    # whether contract b is selected in purchasing feed i in scenario s
    pm.ub = Var(I, S, within=Binary)
    # whether contract d is selected in purchasing feed i in scenario s
    pm.ud = Var(I, S, within=Binary)
    pm.ub1 = Var(I, S, within=Binary)
    pm.ub2 = Var(I, S, within=Binary)
    pm.ud1 = Var(I, S, within=Binary)
    pm.ud2 = Var(I, S, within=Binary)

    # constraints --------------------------------------------------------------

    # Eq. (10)
    def con_f1(m, i):
        return AL[i] * m.lambd[i] <= m.A[i]
    pm.con_f1 = Constraint(I, rule=con_f1)

    # Eq. (10)
    def con_f2(m, i):
        return AU[i] * m.lambd[i] >= m.A[i]
    pm.con_f2 = Constraint(I, rule=con_f2)

    # Eq. (11)
    def con_f3(m, l):
        return SL[l] * m.theta[l] <= m.S[l]
    pm.con_f3 = Constraint(L, rule=con_f3)

    # Eq. (11)
    def con_f4(m, l):
        return SU[l] * m.theta[l] >= m.S[l]
    pm.con_f4 = Constraint(L, rule=con_f4)

    # Eq. (12)
    def con_e2(m, i, s):
        return sum(m.q[i, l, s] * m.y[l, j, s] for l in L for j in J if (i, l) in Tx and (l, j) in Ty) + sum(m.z[i, j, s] for j in J if (i, j) in Tz) <= m.A[i]
    pm.con_e2 = Constraint(I, S, rule=con_e2)

    # Eq. (13)
    def con_e3(m, l, s):
        return sum(m.y[l, j, s] for j in J if (l, j) in Ty) <= m.S[l]
    pm.con_e3 = Constraint(L, S, rule=con_e3)

    # Eq. (14)
    def con_e5(m, j, s):
        return sum(m.y[l, j, s] for l in L if (l, j) in Ty) + sum(m.z[i, j, s] for i in I if (i, j) in Tz) <= DU[j, s]
    pm.con_e5 = Constraint(J, S, rule=con_e5)

    # Eq. (15)
    # TODO check the discrepancy
    def con_e6(m, l, s):
        # # GMS code
        # return sum(m.q[i, l, s] for i in I if (i, l) in Tx) == 1
        # paper
        return sum(m.q[i, l, s] for i in I if (i, l) in Tx) == m.theta[l]
    pm.con_e6 = Constraint(L, S, rule=con_e6)

    # Eq. (16)
    def con_e8(m, j, k, s):
        return PL[j, k] * (sum(m.y[l, j, s] for l in L if (l, j) in Ty) + sum(m.z[i, j, s] for i in I if (i, j) in Tz)) <= sum(C[i, k] * m.z[i, j, s] for i in I if (i, j) in Tz) + sum(C[i, k] * m.q[i, l, s] * m.y[l, j, s] for l in L for i in I if (l, j) in Ty and (i, l) in Tx)
    pm.con_e8 = Constraint(J, K, S, rule=con_e8)

    # Eq. (16)
    def con_e9(m, j, k, s):
        return PU[j, k] * (sum(m.y[l, j, s] for l in L if (l, j) in Ty) + sum(m.z[i, j, s] for i in I if (i, j) in Tz)) >= sum(C[i, k] * m.z[i, j, s] for i in I if (i, j) in Tz) + sum(C[i, k] * m.q[i, l, s] * m.y[l, j, s] for l in L for i in I if (l, j) in Ty and (i, l) in Tx)
    pm.con_e9 = Constraint(J, K, S, rule=con_e9)

    # Eq. (21)
    def con_c1(m, i, s):
        return sum(m.q[i, l, s] * m.y[l, j, s] for l in L for j in J if (i, l) in Tx and (l, j) in Ty) + sum(m.z[i, j, s] for j in J if (i, j) in Tz) == m.Bf[i, s] + m.Bd[i, s] + m.Bb[i, s]
    pm.con_c1 = Constraint(I, S, rule=con_c1)

    # Eq. (22)
    def con_c2(m, i, s):
        return AL[i] * m.uf[i, s] <= m.Bf[i, s]
    pm.con_c2 = Constraint(I, S, rule=con_c2)

    # Eq. (22)
    def con_c3(m, i, s):
        return AU[i] * m.uf[i, s] >= m.Bf[i, s]
    pm.con_c3 = Constraint(I, S, rule=con_c3)

    # Eq. (22)
    def con_c4(m, i, s):
        return AL[i] * m.ud[i, s] <= m.Bd[i, s]
    pm.con_c4 = Constraint(I, S, rule=con_c4)

    # Eq. (22)
    def con_c5(m, i, s):
        return AU[i] * m.ud[i, s] >= m.Bd[i, s]
    pm.con_c5 = Constraint(I, S, rule=con_c5)

    # Eq. (22)
    def con_c6(m, i, s):
        return AL[i] * m.ub[i, s] <= m.Bb[i, s]
    pm.con_c6 = Constraint(I, S, rule=con_c6)

    # Eq. (22)
    def con_c7(m, i, s):
        return AU[i] * m.ub[i, s] >= m.Bb[i, s]
    pm.con_c7 = Constraint(I, S, rule=con_c7)

    # Eq. (23)
    def con_c8(m, i, s):
        return m.uf[i, s] + m.ub[i, s] + m.ud[i, s] <= m.lambd[i]
    pm.con_c8 = Constraint(I, S, rule=con_c8)

    # Eq. (24)
    def con_c9(m, i, s):
        return m.CT[i, s] == m.CTf[i, s] + m.CTb[i, s] + m.CTd[i, s]
    pm.con_c9 = Constraint(I, S, rule=con_c9)

    # Eq. (25)
    def con_c10(m, i, s):
        return m.CTf[i, s] == psi_f[i, s] * m.Bf[i, s]
    pm.con_c10 = Constraint(I, S, rule=con_c10)

    # Eq. (26)
    def con_c11(m, i, s):
        return m.CTd[i, s] == psi_d1[i, s] * m.Bd1[i, s] + psi_d2[i, s] * m.Bd2[i, s]
    pm.con_c11 = Constraint(I, S, rule=con_c11)

    # Eq. (27)
    def con_c12(m, i, s):
        return m.Bd[i, s] == m.Bd1[i, s] + m.Bd2[i, s]
    pm.con_c12 = Constraint(I, S, rule=con_c12)

    # Eq. (28)
    def con_c13(m, i, s):
        return m.Bd1[i, s] == m.Bd11[i, s] + m.Bd12[i, s]
    pm.con_c13 = Constraint(I, S, rule=con_c13)

    # Eq. (29)
    def con_c14(m, i, s):
        return m.Bd11[i, s] <= sigma_d[i] * m.ud1[i, s]
    pm.con_c14 = Constraint(I, S, rule=con_c14)

    # Eq. (30)
    def con_c15(m, i, s):
        return m.Bd12[i, s] == sigma_d[i] * m.ud2[i, s]
    pm.con_c15 = Constraint(I, S, rule=con_c15)

    # Eq. (31)
    def con_c16(m, i, s):
        return m.Bd2[i, s] <= AU[i] * m.ud2[i, s]
    pm.con_c16 = Constraint(I, S, rule=con_c16)

    # Eq. (32)
    def con_c17(m, i, s):
        return m.CTb[i, s] == psi_b1[i, s] * m.Bb1[i, s] + psi_b2[i, s] * m.Bb2[i, s]
    pm.con_c17 = Constraint(I, S, rule=con_c17)

    # Eq. (33)
    def con_c18(m, i, s):
        return m.Bb[i, s] == m.Bb1[i, s] + m.Bb2[i, s]
    pm.con_c18 = Constraint(I, S, rule=con_c18)

    # Eq. (34)
    def con_c19(m, i, s):
        return m.Bb1[i, s] <= sigma_b[i] * m.ub1[i, s]
    pm.con_c19 = Constraint(I, S, rule=con_c19)

    # Eq. (35)
    def con_c20(m, i, s):
        return m.Bb2[i, s] >= sigma_b[i] * m.ub2[i, s]
    pm.con_c20 = Constraint(I, S, rule=con_c20)

    # Eq. (35)
    def con_c21(m, i, s):
        return m.Bb2[i, s] <= AU[i] * m.ub2[i, s]
    pm.con_c21 = Constraint(I, S, rule=con_c21)

    # Eq. (36)
    def con_c22(m, i, s):
        return m.ub1[i, s] + m.ub2[i, s] == m.ub[i, s]
    pm.con_c22 = Constraint(I, S, rule=con_c22)

    # objective ----------------------------------------------------------------

    def obj(m):
        return sum(delta[i] * m.lambd[i] + xi[i] * m.A[i] for i in I) + \
            sum(alpha[l] * m.theta[l] + beta[l] * m.S[l] for l in L) + \
            sum(
                tau[s] * (
                    sum(m.CT[i, s] for i in I) - \
                    sum(
                        d[j, s] * (sum(m.y[l, j, s] for l in L if (l, j) in Ty) + sum(m.z[i, j, s] for i in I if (i, j) in Tz))
                        for j in J
                        )
                    )
                for s in S
                )
    pm.obj = Objective(expr=obj)

    # model transformation -----------------------------------------------------

    builder = StoModelBuilder('pyomo', name='pooling', m_type='MINLP', hint=False)

    scenarios = [1, 2, 3]
    var1_names = ['S', 'A', 'lambd', 'theta']
    con1_names = ['con_f1', 'con_f2', 'con_f3', 'con_f4']

    def _obj(m, s):
        return 1 / 3 * sum(delta[i] * m.lambd[i] + xi[i] * m.A[i] for i in I) + \
            1 / 3 * sum(alpha[l] * m.theta[l] + beta[l] * m.S[l] for l in L) + \
            tau[s] * (
                sum(m.CT[i, s] for i in I) - \
                sum(
                    d[j, s] * (sum(m.y[l, j, s] for l in L if (l, j) in Ty) + sum(m.z[i, j, s] for i in I if (i, j) in Tz))
                    for j in J
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
