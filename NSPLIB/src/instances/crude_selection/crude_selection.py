"""
source
    https://link.springer.com/article/10.1007/s10898-019-00816-8
    https://github.com/bbrunaud/PlasmoAlgorithms.jl/blob/clrootnode/examples/refinery_model/continous/scaled_refinery_continous.gms
    Can Li, Ignacio Grossmann
"""


from pyomo.environ import RangeSet, ConcreteModel, Var, NonNegativeReals, Binary, Constraint, Objective, DataPortal, Param
from ...main import StoModelBuilder
from pathlib import Path


def const_model():
    """Construct the model. This model is first built as a Pyomo model and then
    transferred into StochasticModel.
    """

    pm = ConcreteModel()

    # data portal --------------------------------------------------------------

    data = DataPortal()
    # data.load(filename='src/instances/crude_selection/data.dat')
    script_path = Path(__file__, '..').resolve()
    data.load(filename=str(script_path.joinpath('data.dat')))

    # sets ---------------------------------------------------------------------

    # crudes
    pm.C = RangeSet(1, 10)
    # components
    pm.W = RangeSet(1, 8)
    # reformer input
    pm.RE_IN = RangeSet(1, 2)
    # reformer output
    pm.RE_OUT = RangeSet(1, 6)
    # cracker input
    pm.CR_IN = RangeSet(1, 2)
    # cracker output
    pm.CR_OUT = RangeSet(1, 6)
    # cracker-CGO output
    pm.CR_CGO = RangeSet(1, 3)
    # cracker modes
    pm.CR_MODE = RangeSet(1, 2)
    # isomerization output
    pm.ISO_OUT = RangeSet(1, 4)
    # desulphurization output
    pm.DES_OUT = RangeSet(1, 4)
    # PG98 input
    pm.PG98_IN = RangeSet(1, 6)
    # burn streams
    pm.BURN = RangeSet(1, 3)
    # JPF input
    pm.JPF_IN = RangeSet(1, 3)
    # JPF output
    pm.JPF_OUT = RangeSet(1, 2)
    # AGO input
    pm.AGO_IN = RangeSet(1, 3)
    # products
    pm.P = RangeSet(1, 7)
    # LG input
    pm.LG_IN = RangeSet(1, 5)
    # LG output
    pm.LG_OUT = RangeSet(1, 4)
    # LG properties
    pm.LG_PROP = RangeSet(1, 3)
    # scenarios
    pm.S = RangeSet(1, 5)

    # parameters ---------------------------------------------------------------

    # scalars
    Desulphurisation_capacity = 125
    CDU_capacity = 700
    Reformer95_lower = 5
    Reformer_capacity = 65
    Cracker_capacity = 175
    GranularityOfBarrels = 5000
    LG_sale = 561.6
    LN_sale = 1003
    HF_sale = 637
    ES95_sale = 1194
    PG98_sale = 1231
    JET_sale = 923
    AGO_sale = 907
    CGO_density = 0.95
    Mogas_viscosity = 12.2
    AGO_viscosity = 11.65
    Mogas_Sulphur = 2.1
    AGO_sulphur = 1.68
    Isomerisation_cost = 6
    Reformer95_cost = 2.7
    Reformer100_cost = 3.2
    Cracker_Mogas_cost = 3.2
    Cracker_AGO_cost = 3
    Barrel_lower_bound = 100000
    Barrel_upper_bound = 1500000
    Sulphur_spec = 0.0015
    Desulphurisation_CGO_cost = ((Mogas_Sulphur*109.0909 + 365.4546)/1000)*(0.85/0.159)/CGO_density

    # tables

    pm.Reformer_fraction = Param(pm.RE_IN, pm.RE_OUT, initialize=data.data('Reformer_fraction'))
    pm.Cracker_fraction = Param(pm.CR_IN, pm.CR_OUT, initialize=data.data('Cracker_fraction'))
    pm.Desulphurisation_fraction = Param(pm.C, pm.DES_OUT, initialize=data.data('Desulphurisation_fraction'))
    pm.JPF_fraction = Param(pm.JPF_IN, pm.JPF_OUT, initialize=data.data('JPF_fraction'))
    pm.Crude_yield = Param(pm.C, pm.W, initialize=data.data('Crude_yield'))
    pm.LG_parameters = Param(pm.LG_PROP, pm.LG_IN, initialize=data.data('LG_parameters'))
    pm.Sulphur_GO_data = Param(pm.C, pm.S, initialize=data.data('Sulphur_GO_data'))
    pm.VaccuumResidue_data = Param(pm.C, pm.S, initialize=data.data('VaccuumResidue_data'))

    # indexed parameters

    pm.Isomerisation_fraction = Param(pm.ISO_OUT, initialize=data.data('Isomerisation_fraction'))
    pm.Desulphurisation_fraction2 = Param(pm.DES_OUT, initialize=data.data('Desulphurisation_fraction2'))
    pm.Crude_density = Param(pm.C, initialize=data.data('Crude_density'))
    pm.Sulphur_GO_nominal = Param(pm.C, initialize=data.data('Sulphur_GO_nominal'))
    pm.Crude_price = Param(pm.C, initialize=data.data('Crude_price'))
    pm.Demand_quantity = Param(pm.P, initialize=data.data('Demand_quantity'))
    pm.Density_PG98_input = Param(pm.PG98_IN, initialize=data.data('Density_PG98_input'))
    pm.Density_products = Param(pm.P, initialize=data.data('Density_products'))
    pm.Product_VP = Param(pm.P, initialize=data.data('Product_VP'))
    pm.Product_RON = Param(pm.P, initialize=data.data('Product_RON'))
    pm.Product_MON = Param(pm.P, initialize=data.data('Product_MON'))
    pm.Product_Sulphur = Param(pm.P, initialize=data.data('Product_Sulphur'))
    pm.Import_upper = Param(pm.P, initialize=data.data('Import_upper'))
    pm.RON = Param(pm.PG98_IN, initialize=data.data('RON'))
    pm.MON = Param(pm.PG98_IN, initialize=data.data('MON'))
    pm.VP = Param(pm.PG98_IN, initialize=data.data('VP'))
    pm.HFO_density = Param(pm.C, initialize=data.data('HFO_density'))
    pm.GO_density = Param(pm.C, initialize=data.data('GO_density'))
    pm.Viscosity_HF1 = Param(pm.C, initialize=data.data('Viscosity_HF1'))
    pm.Viscosity_HF3 = Param(pm.C, initialize=data.data('Viscosity_HF3'))
    pm.Viscosity_products = Param(pm.P, initialize=data.data('Viscosity_products'))
    pm.Sulphur_3 = Param(pm.AGO_IN, initialize=data.data('Sulphur_3'))

    # parameters with preprocessing
    BarrelToKT = {}
    Sulphur_GO_stdev = {}
    VaccuumResidue_nominal = {}
    VaccuumResidue_stdev = {}
    for c in pm.C:
        BarrelToKT[c] = (GranularityOfBarrels/6.29)*(pm.Crude_density[c]/1000)
        Sulphur_GO_stdev[c] = 0.1*pm.Sulphur_GO_nominal[c]
        VaccuumResidue_nominal[c] = pm.Crude_yield[c, 8]
        VaccuumResidue_stdev[c] = 0.1 * VaccuumResidue_nominal[c]
    pm.BarrelToKT = Param(pm.C, initialize=BarrelToKT)
    pm.Sulphur_GO_stdev = Param(pm.C, initialize=Sulphur_GO_stdev)
    pm.VaccuumResidue_nominal = Param(pm.C, initialize=VaccuumResidue_nominal)
    pm.VaccuumResidue_stdev = Param(pm.C, initialize=VaccuumResidue_stdev)

    prob = {}
    for s in pm.S:
        prob[s] = 1 / len(pm.S)
    pm.prob = Param(pm.S, initialize=prob)

    Crude_yield_data = {}
    for s in pm.S:
        for c in pm.C:
            for w in pm.W:
                if w == 8:
                    Crude_yield_data[c, w, s] = pm.VaccuumResidue_data[c, s]
                else:
                    Crude_yield_data[c, w, s] = pm.Crude_yield[c, w] / (1 - pm.Crude_yield[c, 8]) * (1 - pm.VaccuumResidue_data[c, s])
    pm.Crude_yield_data = Param(pm.C, pm.W, pm.S, initialize=Crude_yield_data)

    Desulphurisation_cost = {}
    Sulphur_2 = {}
    for c in pm.C:
        for s in pm.S:
            Desulphurisation_cost[c, s] = ( (pm.Sulphur_GO_data[c, s] * 109.0909 + 365.4546) / 1000 ) * (0.85 / 0.159) / pm.GO_density[c]
            Sulphur_2[c, s] = pm.Sulphur_GO_data[c, s] * 0.005
    pm.Desulphurisation_cost = Param(pm.C, pm.S, initialize=Desulphurisation_cost)
    pm.Sulphur_2 = Param(pm.C, pm.S, initialize=Sulphur_2)

    Crude_lower_bound = {}
    Crude_upper_bound = {}
    for c in pm.C:
        Crude_lower_bound[c] = (Barrel_lower_bound / GranularityOfBarrels) * pm.BarrelToKT[c]
        Crude_upper_bound[c] = (Barrel_upper_bound / GranularityOfBarrels) * pm.BarrelToKT[c]
    pm.Crude_lower_bound = Param(pm.C, initialize=Crude_lower_bound)
    pm.Crude_upper_bound = Param(pm.C, initialize=Crude_upper_bound)

    # variables ----------------------------------------------------------------

    # first-stage variables
    pm.pickCrude = Var(pm.C, within=Binary)
    pm.crudeQuantity = Var(pm.C, within=NonNegativeReals)
    for c in pm.C:
        pm.crudeQuantity[c].setub(pm.Crude_upper_bound[c])
    # # manually restrict bounds for unchosen crudes
    # chosen_crudes = [2, 3, 4, 8, 10]
    # unchosen_crudes = [i for i in pm.C if i not in chosen_crudes]
    # for c in unchosen_crudes:
    #     pm.crudeQuantity[c].setub(0)

    # second-stage variables
    pm.flow_Reformer95 = Var(pm.S, within=NonNegativeReals, bounds=(Reformer95_lower, Reformer_capacity))
    pm.flow_Reformer100 = Var(pm.S, within=NonNegativeReals, bounds=(None, Reformer_capacity - Reformer95_lower))
    pm.flow_Cracker_Mogas = Var(pm.S, within=NonNegativeReals, bounds=(None, Cracker_capacity))
    pm.flow_Cracker_AGO = Var(pm.S, within=NonNegativeReals, bounds=(None, Cracker_capacity))
    pm.flow_Isomerisation = Var(pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.flow_Desulphurisation_CGO = Var(pm.S, within=NonNegativeReals, bounds=(None, Cracker_capacity))
    pm.flow_LG_producing = Var(pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.flow_LN_producing = Var(pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.flow_HF_2 = Var(pm.S, within=NonNegativeReals, bounds=(None, Cracker_capacity))
    pm.volume_PG98 = Var(pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity / pm.Density_PG98_input[1]))
    pm.volume_ES95 = Var(pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity / pm.Density_PG98_input[1]))
    pm.volume_HF = Var(pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity / pm.GO_density[7]))
    pm.blin_CDU_LG = Var(pm.LG_OUT, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.blin_Reformer95_LG = Var(pm.LG_OUT, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.blin_Reformer100_LG = Var(pm.LG_OUT, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.blin_Mogas_LG = Var(pm.LG_OUT, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.blin_AGO_LG = Var(pm.LG_OUT, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.blin_Cracker_Mogas = Var(pm.CR_CGO, pm.S, within=NonNegativeReals, bounds=(None, Cracker_capacity))
    pm.blin_Cracker_AGO = Var(pm.CR_CGO, pm.S, within=NonNegativeReals, bounds=(None, Cracker_capacity))
    pm.flow_Desulphurisation_1 = Var(pm.C, pm.S, within=NonNegativeReals, bounds=(None, Desulphurisation_capacity))
    pm.flow_AGO_1 = Var(pm.C, pm.S, within=NonNegativeReals)
    pm.flow_AGO_2 = Var(pm.C, pm.S, within=NonNegativeReals, bounds=(None, Desulphurisation_capacity))
    pm.flow_HF_1 = Var(pm.C, pm.S, within=NonNegativeReals)
    pm.flow_HF_3 = Var(pm.C, pm.S, within=NonNegativeReals)
    pm.flow_Burn = Var(pm.BURN, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.flow_PG98 = Var(pm.PG98_IN, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.flow_ES95 = Var(pm.PG98_IN, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.flow_AGO_3 = Var(pm.AGO_IN, pm.S, within=NonNegativeReals, bounds=(None, Cracker_capacity))
    for s in pm.S:
        pm.flow_AGO_3[1, s].setub(CDU_capacity)
    pm.flow_JPF = Var(pm.JPF_OUT, pm.S, within=NonNegativeReals, bounds=(None, CDU_capacity))
    pm.flow_Import = Var(pm.P, pm.S, within=NonNegativeReals)
    for p in pm.P:
        for s in pm.S:
            pm.flow_Import[p, s].setub(pm.Import_upper[p])
    pm.fraction_LG = Var(pm.LG_IN, pm.S, within=NonNegativeReals, bounds=(None, 1))
    pm.fraction_CGO = Var(pm.CR_MODE, pm.S, within=NonNegativeReals, bounds=(None, 1))
    for c in pm.C:
        for s in pm.S:
            pm.flow_AGO_1[c, s].setub(pm.Crude_upper_bound[c])
            pm.flow_HF_1[c, s].setub(pm.Crude_upper_bound[c])
            pm.flow_HF_3[c, s].setub(pm.Crude_upper_bound[c])

    # constraints --------------------------------------------------------------

    def con_CDU_capacity_bound(m):
        return sum(m.crudeQuantity[c] for c in m.C) <= CDU_capacity
    pm.con_CDU_capacity_bound = Constraint(rule=con_CDU_capacity_bound)

    def con_Crude_selection(m, c):
        return m.crudeQuantity[c] >= m.pickCrude[c] * m.Crude_lower_bound[c]
    pm.con_Crude_selection = Constraint(pm.C, rule=con_Crude_selection)

    def con_Crude_bound(m, c):
        return m.crudeQuantity[c] <= m.pickCrude[c] * m.Crude_upper_bound[c]
    pm.con_Crude_bound = Constraint(pm.C, rule=con_Crude_bound)

    def con_Desulphurisation_capacity_bound(m, s):
        return m.flow_Desulphurisation_CGO[s] + sum(m.flow_Desulphurisation_1[c, s] for c in m. C)  <= Desulphurisation_capacity
    pm.con_Desulphurisation_capacity_bound = Constraint(pm.S, expr=con_Desulphurisation_capacity_bound)

    def con_Mass_balance1(m, s):
        return m.Reformer_fraction[1, 1] * m.flow_Reformer95[s] + \
            m.Reformer_fraction[2, 1] * m.flow_Reformer100[s] + \
            m.Cracker_fraction[1, 1] * m.flow_Cracker_Mogas[s] + \
            m.Cracker_fraction[2, 1] * m.flow_Cracker_AGO[s] + \
            m.Isomerisation_fraction[1] * m.flow_Isomerisation[s] + \
            m.Desulphurisation_fraction2[2] * m.flow_Desulphurisation_CGO[s] - \
            m.flow_Burn[1, s] + sum(m.Crude_yield_data[c, 1, s] * m.crudeQuantity[c] + m.Desulphurisation_fraction[c, 2] * m.flow_Desulphurisation_1[c, s] for c in m.C) \
            == 0
    pm.con_Mass_balance1 = Constraint(pm.S, expr=con_Mass_balance1)

    def con_Mass_balance2(m, s):
        return m.Reformer_fraction[1, 2] * m.flow_Reformer95[s] + \
            m.Reformer_fraction[2, 2] * m.flow_Reformer100[s] + \
            m.Cracker_fraction[1, 2] * m.flow_Cracker_Mogas[s] + \
            m.Cracker_fraction[2, 2] * m.flow_Cracker_AGO[s] - \
            m.flow_LG_producing[s] - m.flow_PG98[1, s] - \
            m.flow_ES95[1, s] - m.flow_Burn[2, s] + \
            sum(m.Crude_yield_data[c, 2, s] * m.crudeQuantity[c] for c in m.C) == 0
    pm.con_Mass_balance2 = Constraint(pm.S, expr=con_Mass_balance2)

    def con_Mass_balance3(m, s):
        return - m.flow_LN_producing[s] - m.flow_Burn[3, s] - m.flow_PG98[3, s] - \
            m.flow_ES95[3, s] - m.flow_Isomerisation[s] - \
            m.flow_JPF[1, s] * m.JPF_fraction[1, 1] - \
            m.flow_JPF[2, s] * m.JPF_fraction[1, 2] + \
            sum(m.Crude_yield_data[c, 3, s] * m.crudeQuantity[c] for c in m.C) == 0
    pm.con_Mass_balance3 = Constraint(pm.S, expr=con_Mass_balance3)

    def con_Mass_balance4(m, s):
        return - m.flow_JPF[1, s] * m.JPF_fraction[2, 1] - \
            m.flow_JPF[2, s] * m.JPF_fraction[2, 2] - \
            m.flow_Reformer95[s] - m.flow_Reformer100[s] + \
            sum(m.Crude_yield_data[c, 4, s] * m.crudeQuantity[c] for c in m.C) == 0
    pm.con_Mass_balance4 = Constraint(pm.S, expr=con_Mass_balance4)

    def con_Mass_balance5(m, s):
        return - m.flow_JPF[1, s] * m.JPF_fraction[3, 1] - \
            m.flow_JPF[2, s] * m.JPF_fraction[3, 2] - m.flow_AGO_3[1, s] + \
            sum(m.Crude_yield_data[c, 5, s] * m.crudeQuantity[c] for c in m.C) == 0
    pm.con_Mass_balance5 = Constraint(pm.S, expr=con_Mass_balance5)

    def con_Mass_balance7(m, s):
        return - m.flow_Cracker_Mogas[s] - m.flow_Cracker_AGO[s] + \
            sum(m.Crude_yield_data[c, 7, s] * m.crudeQuantity[c] for c in m.C) == 0
    pm.con_Mass_balance7 = Constraint(pm.S, expr=con_Mass_balance7)

    def con_GO_balance(m, c, s):
        return - m.flow_AGO_1[c, s] - m.flow_Desulphurisation_1[c, s] - m.flow_HF_3[c, s] + m.Crude_yield_data[c, 6, s] * m.crudeQuantity[c] == 0
    pm.con_GO_balance = Constraint(pm.C, pm.S, expr=con_GO_balance)

    def con_VR_balance(m, c, s):
        return m.Crude_yield_data[c, 8, s] * m.crudeQuantity[c] == m.flow_HF_1[c, s]
    pm.con_VR_balance = Constraint(pm.C, pm.S, expr=con_VR_balance)

    def con_Desulphurisation_balance(m, c, s):
        return m.Desulphurisation_fraction[c, 1] * m.flow_Desulphurisation_1[c, s] == m.flow_AGO_2[c, s]
    pm.con_Desulphurisation_balance = Constraint(pm.C, pm.S, expr=con_Desulphurisation_balance)

    def con_Reformer95_balance(m, s):
        return m.flow_Reformer95[s] * m.Reformer_fraction[1, 3] + m.flow_Reformer100[s] * m.Reformer_fraction[2, 3] == m.flow_PG98[4, s] + m.flow_ES95[4, s]
    pm.con_Reformer95_balance = Constraint(pm.S, expr=con_Reformer95_balance)

    def con_Reformer100_balance(m, s):
        return m.flow_Reformer95[s] * m.Reformer_fraction[1, 4] + m.flow_Reformer100[s] * m.Reformer_fraction[2, 4] == m.flow_PG98[5, s] + m.flow_ES95[5, s]
    pm.con_Reformer100_balance = Constraint(pm.S, expr=con_Reformer100_balance)

    def con_Isomerisation_balance(m, s):
        return m.flow_Isomerisation[s] * m.Isomerisation_fraction[2] == m.flow_PG98[2, s] + m.flow_ES95[2, s]
    pm.con_Isomerisation_balance = Constraint(pm.S, expr=con_Isomerisation_balance)

    def con_CN_balance(m, s):
        return m.flow_Cracker_Mogas[s] * m.Cracker_fraction[1, 3] + m.flow_Cracker_AGO[s] * m.Cracker_fraction[2, 3] == m.flow_PG98[6, s] + m.flow_ES95[6, s]
    pm.con_CN_balance = Constraint(pm.S, expr=con_CN_balance)

    def con_CGO_balance(m, s):
        return m.flow_Cracker_Mogas[s] * m.Cracker_fraction[1, 4] + m.flow_Cracker_AGO[s] * m.Cracker_fraction[2, 4] == m.flow_Desulphurisation_CGO[s] + m.flow_HF_2[s] + m.flow_AGO_3[2, s]
    pm.con_CGO_balance = Constraint(pm.S, expr=con_CGO_balance)

    def con_Desulphurisation_CGO_balance(m, s):
        return m.Desulphurisation_fraction2[1] * m.flow_Desulphurisation_CGO[s] == m.flow_AGO_3[3, s]
    pm.con_Desulphurisation_CGO_balance = Constraint(pm.S, expr=con_Desulphurisation_CGO_balance)

    def con_Demand_constraint1(m, s):
        return m.flow_Import[1, s] + sum(m.flow_PG98[i, s] for i in m.PG98_IN) >= m.Demand_quantity[1]
    pm.con_Demand_constraint1 = Constraint(pm.S, expr=con_Demand_constraint1)

    def con_Demand_constraint2(m, s):
        return m.flow_Import[2, s] + sum(m.flow_ES95[PG98_in, s] for PG98_in in m.PG98_IN) >= m.Demand_quantity[2]
    pm.con_Demand_constraint2 = Constraint(pm.S, expr=con_Demand_constraint2)

    def con_Demand_constraint3(m, s):
        return m.flow_Import[3, s] + sum(m.flow_JPF[JPF_out, s] for JPF_out in m.JPF_OUT) >= m.Demand_quantity[3]
    pm.con_Demand_constraint3 = Constraint(pm.S, expr=con_Demand_constraint3)

    def con_Demand_constraint4(m, s):
        return m.flow_Import[4, s] + sum(m.flow_AGO_3[AGO_in, s] for AGO_in in m.AGO_IN) + sum(m.flow_AGO_1[c, s] + m.flow_AGO_2[c, s] for c in m.C) >= m.Demand_quantity[4]
    pm.con_Demand_constraint4 = Constraint(pm.S, expr=con_Demand_constraint4)

    def con_Demand_constraint5(m, s):
        return m.flow_Import[5, s] + m.flow_HF_2[s] + sum(m.flow_HF_1[c, s] + m.flow_HF_3[c, s] for c in m.C) >= m.Demand_quantity[5]
    pm.con_Demand_constraint5 = Constraint(pm.S, expr=con_Demand_constraint5)

    def con_Demand_constraint6(m, s):
        return m.flow_Import[6, s] + m.flow_LG_producing[s] >= m.Demand_quantity[6]
    pm.con_Demand_constraint6 = Constraint(pm.S, expr=con_Demand_constraint6)

    def con_Demand_constraint7(m, s):
        return m.flow_Import[7, s] + m.flow_LN_producing[s] >= m.Demand_quantity[7]
    pm.con_Demand_constraint7 = Constraint(pm.S, expr=con_Demand_constraint7)

    def con_PG98_volume_def(m, s):
        return m.flow_Import[1, s] / m.Density_products[1] + sum(m.flow_PG98[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) == m.volume_PG98[s]
    pm.con_PG98_volume_def = Constraint(pm.S, expr=con_PG98_volume_def)

    def con_ES95_volume_def(m, s):
        return m.flow_Import[2, s] / m.Density_products[2] + sum(m.flow_ES95[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) == m.volume_ES95[s]
    pm.con_ES95_volume_def = Constraint(pm.S, expr=con_ES95_volume_def)

    def con_Butane95_constraint(m, s):
        return m.flow_ES95[1, s] / m.Density_PG98_input[1] + 0.03 * m.flow_Import[2, s] / m.Density_products[2] <= 0.05 * m.volume_ES95[s]
    pm.con_Butane95_constraint = Constraint(pm.S, expr=con_Butane95_constraint)

    def con_Butane98_constraint(m, s):
        return m.flow_PG98[1, s] / m.Density_PG98_input[1] + 0.03 * m.flow_Import[1, s] / m.Density_products[2] <= 0.05 * m.volume_PG98[s]
    pm.con_Butane98_constraint = Constraint(pm.S, expr=con_Butane98_constraint)

    def con_blincon_CDU_LG1(m, s):
        return m.blin_CDU_LG[1, s] == m.fraction_LG[1, s] * m.flow_ES95[1, s]
    pm.con_blincon_CDU_LG1 = Constraint(pm.S, expr=con_blincon_CDU_LG1)

    def con_blincon_CDU_LG2(m, s):
        return m.blin_CDU_LG[2, s] == m.fraction_LG[1, s] * m.flow_PG98[1, s]
    pm.con_blincon_CDU_LG2 = Constraint(pm.S, expr=con_blincon_CDU_LG2)

    def con_blincon_CDU_LG3(m, s):
        return m.blin_CDU_LG[3, s] == m.fraction_LG[1, s] * m.flow_Burn[2, s]
    pm.con_blincon_CDU_LG3 = Constraint(pm.S, expr=con_blincon_CDU_LG3)

    def con_blincon_CDU_LG4(m, s):
        return m.blin_CDU_LG[4, s] == m.fraction_LG[1, s] * m.flow_LG_producing[s]
    pm.con_blincon_CDU_LG4 = Constraint(pm.S, expr=con_blincon_CDU_LG4)

    def con_blincon_Reformer95_LG1(m, s):
        return m.blin_Reformer95_LG[1, s] == m.fraction_LG[2, s] * m.flow_ES95[1, s]
    pm.con_blincon_Reformer95_LG1 = Constraint(pm.S, expr=con_blincon_Reformer95_LG1)

    def con_blincon_Reformer95_LG2(m, s):
        return m.blin_Reformer95_LG[2, s] == m.fraction_LG[2, s] * m.flow_PG98[1, s]
    pm.con_blincon_Reformer95_LG2 = Constraint(pm.S, expr=con_blincon_Reformer95_LG2)

    def con_blincon_Reformer95_LG3(m, s):
        return m.blin_Reformer95_LG[3, s] == m.fraction_LG[2, s] * m.flow_Burn[2, s]
    pm.con_blincon_Reformer95_LG3 = Constraint(pm.S, expr=con_blincon_Reformer95_LG3)

    def con_blincon_Reformer95_LG4(m, s):
        return m.blin_Reformer95_LG[4, s] == m.fraction_LG[2, s] * m.flow_LG_producing[s]
    pm.con_blincon_Reformer95_LG4 = Constraint(pm.S, expr=con_blincon_Reformer95_LG4)

    def con_blincon_Reformer100_LG1(m, s):
        return m.blin_Reformer100_LG[1, s] == m.fraction_LG[3, s] * m.flow_ES95[1, s]
    pm.con_blincon_Reformer100_LG1 = Constraint(pm.S, expr=con_blincon_Reformer100_LG1)

    def con_blincon_Reformer100_LG2(m, s):
        return m.blin_Reformer100_LG[2, s] == m.fraction_LG[3, s] * m.flow_PG98[1, s]
    pm.con_blincon_Reformer100_LG2 = Constraint(pm.S, expr=con_blincon_Reformer100_LG2)

    def con_blincon_Reformer100_LG3(m, s):
        return m.blin_Reformer100_LG[3, s] == m.fraction_LG[3, s] * m.flow_Burn[2, s]
    pm.con_blincon_Reformer100_LG3 = Constraint(pm.S, expr=con_blincon_Reformer100_LG3)

    def con_blincon_Reformer100_LG4(m, s):
        return m.blin_Reformer100_LG[4, s] == m.fraction_LG[3, s] * m.flow_LG_producing[s]
    pm.con_blincon_Reformer100_LG4 = Constraint(pm.S, expr=con_blincon_Reformer100_LG4)

    def con_blincon_Mogas_LG1(m, s):
        return m.blin_Mogas_LG[1, s] == m.fraction_LG[4, s] * m.flow_ES95[1, s]
    pm.con_blincon_Mogas_LG1 = Constraint(pm.S, expr=con_blincon_Mogas_LG1)

    def con_blincon_Mogas_LG2(m, s):
        return m.blin_Mogas_LG[2, s] == m.fraction_LG[4, s] * m.flow_PG98[1, s]
    pm.con_blincon_Mogas_LG2 = Constraint(pm.S, expr=con_blincon_Mogas_LG2)

    def con_blincon_Mogas_LG3(m, s):
        return m.blin_Mogas_LG[3, s] == m.fraction_LG[4, s] * m.flow_Burn[2, s]
    pm.con_blincon_Mogas_LG3 = Constraint(pm.S, expr=con_blincon_Mogas_LG3)

    def con_blincon_Mogas_LG4(m, s):
        return m.blin_Mogas_LG[4, s] == m.fraction_LG[4, s] * m.flow_LG_producing[s]
    pm.con_blincon_Mogas_LG4 = Constraint(pm.S, expr=con_blincon_Mogas_LG4)

    def con_blincon_AGO_LG1(m, s):
        return m.blin_AGO_LG[1, s] == m.fraction_LG[5, s] * m.flow_ES95[1, s]
    pm.con_blincon_AGO_LG1 = Constraint(pm.S, expr=con_blincon_AGO_LG1)

    def con_blincon_AGO_LG2(m, s):
        return m.blin_AGO_LG[2, s] == m.fraction_LG[5, s] * m.flow_PG98[1, s]
    pm.con_blincon_AGO_LG2 = Constraint(pm.S, expr=con_blincon_AGO_LG2)

    def con_blincon_AGO_LG3(m, s):
        return m.blin_AGO_LG[3, s] == m.fraction_LG[5, s] * m.flow_Burn[2, s]
    pm.con_blincon_AGO_LG3 = Constraint(pm.S, expr=con_blincon_AGO_LG3)

    def con_blincon_AGO_LG4(m, s):
        return m.blin_AGO_LG[4, s] == m.fraction_LG[5, s] * m.flow_LG_producing[s]
    pm.con_blincon_AGO_LG4 = Constraint(pm.S, expr=con_blincon_AGO_LG4)

    def con_LG_balance(m, s):
        return sum(m.blin_CDU_LG[LG_out, s] for LG_out in m.LG_OUT) == sum(m.Crude_yield_data[c, 2, s] * m.crudeQuantity[c] for c in m.C)
    pm.con_LG_balance = Constraint(pm.S, expr=con_LG_balance)

    def con_Reformer95_LG_balance(m, s):
        return m.flow_Reformer95[s] * m.Reformer_fraction[1, 2] == sum(m.blin_Reformer95_LG[LG_out, s] for LG_out in m.LG_OUT)
    pm.con_Reformer95_LG_balance = Constraint(pm.S, expr=con_Reformer95_LG_balance)

    def con_Reformer100_LG_balance(m, s):
        return m.flow_Reformer100[s] * m.Reformer_fraction[2, 2] == sum(m.blin_Reformer100_LG[LG_out, s] for LG_out in m.LG_OUT)
    pm.con_Reformer100_LG_balance = Constraint(pm.S, expr=con_Reformer100_LG_balance)

    def con_Cracker_Mogas_LG_balance(m, s):
        return m.flow_Cracker_Mogas[s] * m.Cracker_fraction[1, 2] == sum(m.blin_Mogas_LG[LG_out, s] for LG_out in m.LG_OUT)
    pm.con_Cracker_Mogas_LG_balance = Constraint(pm.S, expr=con_Cracker_Mogas_LG_balance)

    def con_Cracker_AGO_LG_balance(m, s):
        return m.flow_Cracker_AGO[s] * m.Cracker_fraction[2, 2] == sum(m.blin_AGO_LG[LG_out, s] for LG_out in m.LG_OUT)
    pm.con_Cracker_AGO_LG_balance = Constraint(pm.S, expr=con_Cracker_AGO_LG_balance)

    def con_pq_ES95_constraint(m, s):
        return m.blin_CDU_LG[1, s] + m.blin_Reformer95_LG[1, s] + m.blin_Reformer100_LG[1, s] + m.blin_Mogas_LG[1, s] + m.blin_AGO_LG[1, s] == m.flow_ES95[1, s]
    pm.con_pq_ES95_constraint = Constraint(pm.S, expr=con_pq_ES95_constraint)

    def con_pq_PG98_constraint(m, s):
        return m.blin_CDU_LG[2, s] + m.blin_Reformer95_LG[2, s] + m.blin_Reformer100_LG[2, s] + m.blin_Mogas_LG[2, s] + m.blin_AGO_LG[2, s] == m.flow_PG98[1, s]
    pm.con_pq_PG98_constraint = Constraint(pm.S, expr=con_pq_PG98_constraint)

    def con_pq_burn_constraint(m, s):
        return m.blin_CDU_LG[3, s] + m.blin_Reformer95_LG[3, s] + m.blin_Reformer100_LG[3, s] + m.blin_Mogas_LG[3, s] + m.blin_AGO_LG[3, s] == m.flow_Burn[2, s]
    pm.con_pq_burn_constraint = Constraint(pm.S, expr=con_pq_burn_constraint)

    def con_pq_demand_constraint(m, s):
        return m.blin_CDU_LG[4, s] + m.blin_Reformer95_LG[4, s] + m.blin_Reformer100_LG[4, s] + m.blin_Mogas_LG[4, s] + m.blin_AGO_LG[4, s] == m.flow_LG_producing[s]
    pm.con_pq_demand_constraint = Constraint(pm.S, expr=con_pq_demand_constraint)

    def con_LG_split_balance(m, s):
        return sum(m.fraction_LG[LG_in, s] for LG_in in m.LG_IN) == 1
    pm.con_LG_split_balance = Constraint(pm.S, expr=con_LG_split_balance)

    def con_VP_ES95_lower(m, s):
        return - 0.45 * m.volume_ES95[s] + m.flow_Import[2, s] * m.Product_VP[2] / m.Density_products[2] + \
            sum(m.VP[PG98_in] * m.flow_ES95[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) + \
                m.LG_parameters[1, 1] * m.blin_CDU_LG[1, s] / m.Density_PG98_input[1] + \
                m.LG_parameters[1, 2] * m.blin_Reformer95_LG[1, s] / m.Density_PG98_input[1] + \
                m.LG_parameters[1, 3] * m.blin_Reformer100_LG[1, s] / m.Density_PG98_input[1] + \
                m.LG_parameters[1, 4] * m.blin_Mogas_LG[1, s] / m.Density_PG98_input[1] + \
                m.LG_parameters[1, 5] * m.blin_AGO_LG[1, s] / m.Density_PG98_input[1] >= 0
    pm.con_VP_ES95_lower = Constraint(pm.S, expr=con_VP_ES95_lower)

    def con_VP_ES95_upper(m, s):
        return - 0.80 * m.volume_ES95[s] + \
            m.flow_Import[2, s] * m.Product_VP[2] / m.Density_products[2] + \
            sum(m.VP[PG98_in] * m.flow_ES95[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) + \
            m.LG_parameters[1, 1] * m.blin_CDU_LG[1, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 2] * m.blin_Reformer95_LG[1, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 3] * m.blin_Reformer100_LG[1, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 4] * m.blin_Mogas_LG[1, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 5] * m.blin_AGO_LG[1, s] / m.Density_PG98_input[1] <= 0
    pm.con_VP_ES95_upper = Constraint(pm.S, expr=con_VP_ES95_upper)

    def con_VP_PG98_lower(m, s):
        return - 0.50 * m.volume_PG98[s] + m.flow_Import[1, s] * m.Product_VP[1] / m.Density_products[1] + \
            sum(m.VP[PG98_in] * m.flow_PG98[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) + \
            m.LG_parameters[1, 1] * m.blin_CDU_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 2] * m.blin_Reformer95_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 3] * m.blin_Reformer100_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 4] * m.blin_Mogas_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 5] * m.blin_AGO_LG[2, s] / m.Density_PG98_input[1] >= 0
    pm.con_VP_PG98_lower = Constraint(pm.S, expr=con_VP_PG98_lower)

    def con_VP_PG98_upper(m, s):
        return - 0.86 * m.volume_PG98[s] + m.flow_Import[1, s] * m.Product_VP[1] / m.Density_products[1] + \
            sum(m.VP[PG98_in] * m.flow_PG98[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) + \
            m.LG_parameters[1, 1] * m.blin_CDU_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 2] * m.blin_Reformer95_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 3] * m.blin_Reformer100_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 4] * m.blin_Mogas_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[1, 5] * m.blin_AGO_LG[2, s] / m.Density_PG98_input[1] <= 0
    pm.con_VP_PG98_upper = Constraint(pm.S, expr=con_VP_PG98_upper)

    def con_RON_PG98(m, s):
        return - 98 * m.volume_PG98[s] + m.flow_Import[1, s] * m.Product_RON[1] / m.Density_products[1] + \
            sum(m.RON[PG98_in] * m.flow_PG98[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) + \
            m.LG_parameters[2, 1] * m.blin_CDU_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[2, 2] * m.blin_Reformer95_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[2, 3] * m.blin_Reformer100_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[2, 4] * m.blin_Mogas_LG[2, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[2, 5] * m.blin_AGO_LG[2, s] / m.Density_PG98_input[1] >= 0
    pm.con_RON_PG98 = Constraint(pm.S, expr=con_RON_PG98)

    def con_RON_ES95(m, s):
        return - 95 * m.volume_ES95[s] + m.flow_Import[2, s] * m.Product_RON[2] / m.Density_products[2] + \
            sum(m.RON[PG98_in] * m.flow_ES95[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN)+ \
            m.LG_parameters[2, 1] * m.blin_CDU_LG[1, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[2, 2] * m.blin_Reformer95_LG[1, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[2, 3] * m.blin_Reformer100_LG[1, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[2, 4] * m.blin_Mogas_LG[1, s] / m.Density_PG98_input[1] + \
            m.LG_parameters[2, 5] * m.blin_AGO_LG[1, s] / m.Density_PG98_input[1] >= 0
    pm.con_RON_ES95 = Constraint(pm.S, expr=con_RON_ES95)

    def con_Sensitivity_PG98(m, s):
        return - 10 * m.volume_PG98[s] + \
            m.flow_Import[1, s] * (m.Product_RON[1] - m.Product_MON[1]) / m.Density_products[1] + \
            sum((m.RON[PG98_in] - m.MON[PG98_in]) * m.flow_PG98[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) + \
            (m.LG_parameters[2, 1] - m.LG_parameters[3, 1]) * m.blin_CDU_LG[2, s] / m.Density_PG98_input[1] + \
            (m.LG_parameters[2, 2] - m.LG_parameters[3, 2]) * m.blin_Reformer95_LG[2, s] / m.Density_PG98_input[1] + \
            (m.LG_parameters[2, 3] - m.LG_parameters[3, 3]) * m.blin_Reformer100_LG[2, s] / m.Density_PG98_input[1] + \
            (m.LG_parameters[2, 4] - m.LG_parameters[3, 4]) * m.blin_Mogas_LG[2, s] / m.Density_PG98_input[1] + \
            (m.LG_parameters[2, 5] - m.LG_parameters[3, 5]) * m.blin_AGO_LG[2, s] / m.Density_PG98_input[1] <= 0
    pm.con_Sensitivity_PG98 = Constraint(pm.S, expr=con_Sensitivity_PG98)

    def con_Sensitivity_ES95(m, s):
        return - 10 * m.volume_ES95[s] + \
            m.flow_Import[2, s] * (m.Product_RON[2] - m.Product_MON[2]) / m.Density_products[2] + \
            sum((m.RON[PG98_in] - m.MON[PG98_in]) * m.flow_ES95[PG98_in, s] / m.Density_PG98_input[PG98_in] for PG98_in in m.PG98_IN) + \
            (m.LG_parameters[2, 1] - m.LG_parameters[3, 1]) * m.blin_CDU_LG[1, s] / m.Density_PG98_input[1] + \
            (m.LG_parameters[2, 2] - m.LG_parameters[3, 2]) * m.blin_Reformer95_LG[1, s] / m.Density_PG98_input[1] + \
            (m.LG_parameters[2, 3] - m.LG_parameters[3, 3]) * m.blin_Reformer100_LG[1, s] / m.Density_PG98_input[1] + \
            (m.LG_parameters[2, 4] - m.LG_parameters[3, 4]) * m.blin_Mogas_LG[1, s] / m.Density_PG98_input[1] + \
            (m.LG_parameters[2, 5] - m.LG_parameters[3, 5]) * m.blin_AGO_LG[1, s] / m.Density_PG98_input[1] <= 0
    pm.con_Sensitivity_ES95 = Constraint(pm.S, expr=con_Sensitivity_ES95)

    def con_blincon_Cracker_Mogas1(m, s):
        return m.blin_Cracker_Mogas[1, s] == m.fraction_CGO[1, s] * m.flow_AGO_3[2, s]
    pm.con_blincon_Cracker_Mogas1 = Constraint(pm.S, expr=con_blincon_Cracker_Mogas1)

    def con_blincon_Cracker_Mogas2(m, s):
        return m.blin_Cracker_Mogas[2, s] == m.fraction_CGO[1, s] * m.flow_HF_2[s]
    pm.con_blincon_Cracker_Mogas2 = Constraint(pm.S, expr=con_blincon_Cracker_Mogas2)

    def con_blincon_Cracker_Mogas3(m, s):
        return m.blin_Cracker_Mogas[3, s] == m.fraction_CGO[1, s] * m.flow_Desulphurisation_CGO[s]
    pm.con_blincon_Cracker_Mogas3 = Constraint(pm.S, expr=con_blincon_Cracker_Mogas3)

    def con_blincon_Cracker_AGO1(m, s):
        return m.blin_Cracker_AGO[1, s] == m.fraction_CGO[2, s] * m.flow_AGO_3[2, s]
    pm.con_blincon_Cracker_AGO1 = Constraint(pm.S, expr=con_blincon_Cracker_AGO1)

    def con_blincon_Cracker_AGO2(m, s):
        return m.blin_Cracker_AGO[2, s] == m.fraction_CGO[2, s] * m.flow_HF_2[s]
    pm.con_blincon_Cracker_AGO2 = Constraint(pm.S, expr=con_blincon_Cracker_AGO2)

    def con_blincon_Cracker_AGO3(m, s):
        return m.blin_Cracker_AGO[3, s] == m.fraction_CGO[2, s] * m.flow_Desulphurisation_CGO[s]
    pm.con_blincon_Cracker_AGO3 = Constraint(pm.S, expr=con_blincon_Cracker_AGO3)

    def con_Cracker_Mogas_CGO_balance(m, s):
        return m.blin_Cracker_Mogas[1, s] + m.blin_Cracker_Mogas[2, s] + m.blin_Cracker_Mogas[3, s] == m.flow_Cracker_Mogas[s] * m.Cracker_fraction[1, 4]
    pm.con_Cracker_Mogas_CGO_balance = Constraint(pm.S, expr=con_Cracker_Mogas_CGO_balance)

    def con_Cracker_AGO_CGO_balance(m, s):
        return m.blin_Cracker_AGO[1, s] + m.blin_Cracker_AGO[2, s] + m.blin_Cracker_AGO[3, s] == m.flow_Cracker_AGO[s] * m.Cracker_fraction[2, 4]
    pm.con_Cracker_AGO_CGO_balance = Constraint(pm.S, expr=con_Cracker_AGO_CGO_balance)

    def con_CGO_split_balance(m, s):
        return sum(m.fraction_CGO[Cr_mode, s] for Cr_mode in m.CR_MODE) == 1
    pm.con_CGO_split_balance = Constraint(pm.S, expr=con_CGO_split_balance)

    def con_pq_AGO_constraint(m, s):
        return m.blin_Cracker_Mogas[1, s] + m.blin_Cracker_AGO[1, s] == m.flow_AGO_3[2, s]
    pm.con_pq_AGO_constraint = Constraint(pm.S, expr=con_pq_AGO_constraint)

    def con_pq_HF_constraint(m, s):
        return m.blin_Cracker_Mogas[2, s] + m.blin_Cracker_AGO[2, s] == m.flow_HF_2[s]
    pm.con_pq_HF_constraint = Constraint(pm.S, expr=con_pq_HF_constraint)

    def con_pq_Desulphurisation_constraint(m, s):
        return m.blin_Cracker_Mogas[3, s] + m.blin_Cracker_AGO[3, s] == m.flow_Desulphurisation_CGO[s]
    pm.con_pq_Desulphurisation_constraint = Constraint(pm.S, expr=con_pq_Desulphurisation_constraint)

    def con_HF_volume_def(m, s):
        return - m.volume_HF[s] + m.flow_Import[5, s] / m.Density_products[5] + \
            m.flow_HF_2[s] / CGO_density + \
            sum(m.flow_HF_1[c, s] / m.HFO_density[c] + m.flow_HF_3[c, s] / m.GO_density[c] for c in m.C) == 0
    pm.con_HF_volume_def = Constraint(pm.S, expr=con_HF_volume_def)

    def con_HF_viscosity_lower(m, s):
        return m.flow_Import[5, s] * m.Viscosity_products[5] / m.Density_products[5] + \
            sum(m.flow_HF_1[c, s] * m.Viscosity_HF1[c] / m.HFO_density[c] + m.flow_HF_3[c, s] * m.Viscosity_HF3[c] / m.GO_density[c] for c in m.C) + \
            (m.blin_Cracker_Mogas[2, s] * Mogas_viscosity + m.blin_Cracker_AGO[2, s] * AGO_viscosity) / CGO_density - 30 * m.volume_HF[s] >= 0
    pm.con_HF_viscosity_lower = Constraint(pm.S, expr=con_HF_viscosity_lower)

    def con_HF_viscosity_upper(m, s):
        return m.flow_Import[5, s] * m.Viscosity_products[5] / m.Density_products[5] + \
            sum(m.flow_HF_1[c, s] * m.Viscosity_HF1[c] / m.HFO_density[c] + m.flow_HF_3[c, s] * m.Viscosity_HF3[c] / m.GO_density[c] for c in m.C) + \
            (m.blin_Cracker_Mogas[2, s] * Mogas_viscosity + m.blin_Cracker_AGO[2, s] * AGO_viscosity) / CGO_density - 33 * m.volume_HF[s] <= 0
    pm.con_HF_viscosity_upper = Constraint(pm.S, expr=con_HF_viscosity_upper)

    def con_AGO_sulphur_balance(m, s):
        return m.flow_Import[4, s] * m.Product_Sulphur[4] - Sulphur_spec * m.flow_Import[4, s] + \
            sum((m.Sulphur_GO_data[c, s] - Sulphur_spec) * m.flow_AGO_1[c, s] + (m.Sulphur_2[c, s] - Sulphur_spec) * m.flow_AGO_2[c, s] for c in m.C) + \
            m.flow_AGO_3[1, s] * (m.Sulphur_3[1] - Sulphur_spec) + \
            m.blin_Cracker_AGO[1, s] * (AGO_sulphur - Sulphur_spec) + \
            m.blin_Cracker_Mogas[1, s] * (Mogas_Sulphur - Sulphur_spec) + \
            m.blin_Cracker_AGO[3, s] * AGO_sulphur * 0.005 + \
            m.blin_Cracker_Mogas[3, s] * Mogas_Sulphur * 0.005 - \
            Sulphur_spec * m.flow_AGO_3[3, s] <= 0
    pm.con_AGO_sulphur_balance = Constraint(pm.S, expr=con_AGO_sulphur_balance)

    def con_Refinery_Fuel(m, s):
        return 1.3 * m.flow_Burn[1, s] + 1.2 * m.flow_Burn[2, s] + \
            1.1 * m.flow_Burn[3, s] - m.flow_Reformer95[s] * m.Reformer_fraction[1, 5] - \
            m.flow_Reformer100[s] * m.Reformer_fraction[2, 5] - \
            m.flow_Cracker_Mogas[s] * m.Cracker_fraction[1, 5] - \
            m.flow_Cracker_AGO[s] * m.Cracker_fraction[2, 5] - \
            m.flow_Isomerisation[s] * m.Isomerisation_fraction[3] - \
            m.flow_Desulphurisation_CGO[s] * m.Desulphurisation_fraction2[3] - \
            15.2 - sum(0.018 * m.crudeQuantity[c] + m.flow_Desulphurisation_1[c, s] * m.Desulphurisation_fraction[c, 3] for c in m.C) >= 0
    pm.con_Refinery_Fuel = Constraint(pm.S, expr=con_Refinery_Fuel)

    def con_Cracker_capacity_bound(m, s):
        return m.flow_Cracker_Mogas[s] + m.flow_Cracker_AGO[s] <= Cracker_capacity
    pm.con_Cracker_capacity_bound = Constraint(pm.S, expr=con_Cracker_capacity_bound)

    def con_Reformer_capacity_bound(m, s):
        return m.flow_Reformer95[s] + m.flow_Reformer100[s] <= Reformer_capacity
    pm.con_Reformer_capacity_bound = Constraint(pm.S, expr=con_Reformer_capacity_bound)

    # objective ----------------------------------------------------------------

    def obj(m):
        return sum(
            m.prob[s] * (
                Cracker_Mogas_cost * m.flow_Cracker_Mogas[s] +
                Cracker_AGO_cost * m.flow_Cracker_AGO[s] +
                Reformer95_cost * m.flow_Reformer95[s] +
                Reformer100_cost * m.flow_Reformer100[s] +
                Isomerisation_cost * m.flow_Isomerisation[s] +
                Desulphurisation_CGO_cost * m.flow_Desulphurisation_CGO[s] -
                LG_sale * m.flow_LG_producing[s] -
                LN_sale * m.flow_LN_producing[s] -
                HF_sale * m.flow_HF_2[s] +
                sum(
                    m.Desulphurisation_cost[c, s] * m.flow_Desulphurisation_1[c, s] -
                    AGO_sale * m.flow_AGO_1[c, s] -
                    AGO_sale * m.flow_AGO_2[c, s] -
                    HF_sale * m.flow_HF_1[c, s] -
                    HF_sale * m.flow_HF_3[c, s] +
                    (m.crudeQuantity[c] / 1000 / m.BarrelToKT[c] * GranularityOfBarrels) * (m.Crude_price[c] + 1)
                    for c in m.C
                ) -
                sum(
                    PG98_sale * m.flow_PG98[PG98_in, s] +
                    ES95_sale * m.flow_ES95[PG98_in, s]
                    for PG98_in in m.PG98_IN
                ) -
                sum(
                    JET_sale * m.flow_JPF[i, s] for i in m.JPF_OUT
                ) -
                sum(
                    AGO_sale * m.flow_AGO_3[i, s] for i in m.AGO_IN
                )
            )
            for s in m.S
            )
    pm.obj = Objective(expr=obj)

    # model transformation -----------------------------------------------------

    builder = StoModelBuilder('pyomo', name='crude_selection', m_type='MINLP', hint=False)

    scenarios = list(pm.S)
    var1_names = ['pickCrude', 'crudeQuantity']
    con1_names = ['con_CDU_capacity_bound', 'con_Crude_bound', 'con_Crude_selection']

    def _obj(m, s):
        return \
            m.prob[s] * (
                Cracker_Mogas_cost * m.flow_Cracker_Mogas[s] +
                Cracker_AGO_cost * m.flow_Cracker_AGO[s] +
                Reformer95_cost * m.flow_Reformer95[s] +
                Reformer100_cost * m.flow_Reformer100[s] +
                Isomerisation_cost * m.flow_Isomerisation[s] +
                Desulphurisation_CGO_cost * m.flow_Desulphurisation_CGO[s] -
                LG_sale * m.flow_LG_producing[s] -
                LN_sale * m.flow_LN_producing[s] -
                HF_sale * m.flow_HF_2[s] +
                sum(
                    m.Desulphurisation_cost[c, s] * m.flow_Desulphurisation_1[c, s] -
                    AGO_sale * m.flow_AGO_1[c, s] -
                    AGO_sale * m.flow_AGO_2[c, s] -
                    HF_sale * m.flow_HF_1[c, s] -
                    HF_sale * m.flow_HF_3[c, s] +
                    (m.crudeQuantity[c] / 1000 / m.BarrelToKT[c] * GranularityOfBarrels) * (m.Crude_price[c] + 1)
                    for c in m.C
                ) -
                sum(
                    PG98_sale * m.flow_PG98[PG98_in, s] +
                    ES95_sale * m.flow_ES95[PG98_in, s]
                    for PG98_in in m.PG98_IN
                ) -
                sum(
                    JET_sale * m.flow_JPF[i, s] for i in m.JPF_OUT
                ) -
                sum(
                    AGO_sale * m.flow_AGO_3[i, s] for i in m.AGO_IN
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
