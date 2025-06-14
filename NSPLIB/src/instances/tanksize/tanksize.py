
from pyomo.environ import *
import numpy as np
from ...main import StoModelBuilder
import numpy as np
def cycleTime_constraint(model, s):
    return model.cycleTime[s] == sum(model.campaignDuration[n, s] + sum(model.CampaignSetupTime[p] * model.assignProductToCampaign[p, n, s] for p in model.p) for n in model.n)
def unique_product_rule(model, n, s):
    return sum(model.assignProductToCampaign[p, n, s] for p in model.p) <= 1    
def material_balance_rule(model, p, n, s):
    if n == model.n.last():
        return Constraint.Skip
    else:
        return model.productInventory[p, n+1, s] == model.productInventory[p, n, s] + model.amtProductInCampaign[p, n, s] - model.DemandPerDay[p, s] * (model.campaignDuration[n, s] + sum(model.CampaignSetupTime[pp] * model.assignProductToCampaign[pp, n, s] for pp in model.p))

def tank_capacity_rule(model, p, n, s):
    return model.productInventory[p, n, s] <= model.productTankSize[p]
def production_upper_bound1_rule(model, p, n, s):
    return model.amtProductInCampaign[p, n, s] <= model.MaxProductionRate[p] * model.ProductionLengts_upper[p] * model.assignProductToCampaign[p, n, s]
def production_lower_bound1_rule(model, p, n, s):
    return model.amtProductInCampaign[p, n, s] >= model.MinProductionRate[p] * model.ProductionLengts_lower[p] * model.assignProductToCampaign[p, n, s]
def production_upper_bound2_rule(model, p, n, s):
    return model.amtProductInCampaign[p, n, s] <= model.MaxProductionRate[p] * model.campaignDuration[n, s]
def production_lower_bound2_rule(model, p, n, s):
    return model.amtProductInCampaign[p, n, s] >= model.MinProductionRate[p] * (model.campaignDuration[n, s] - model.ProductionLengts_upper[p] * (1 - model.assignProductToCampaign[p, n, s]))
def campaign_upper_bound_rule(model, n, s):
    return model.campaignDuration[n, s] <= sum(model.ProductionLengts_upper[p] * model.assignProductToCampaign[p, n, s] for p in model.p)
def campaign_lower_bound_rule(model, n, s):
    return model.campaignDuration[n, s] >= sum(model.ProductionLengts_lower[p] * model.assignProductToCampaign[p, n, s] for p in model.p)
def campaign_lengts_def_rule(model, n, s):
    return model.campaignLengts[n, s] == model.campaignDuration[n, s] + sum(model.CampaignSetupTime[pp] * model.assignProductToCampaign[pp, n, s] for pp in model.p)
def campaign_setup_cost_con_rule(model, s):
    return model.setupCost[s] == sum(model.CampaignSetupCost[p] * model.assignProductToCampaign[p, n, s] for p in model.p for n in model.n)
def campaign_storage_cost_rule(model, s):
    return model.variableCost[s] == sum(model.CampaignVariableCost[p] * model.auxiliaryVariable[p, n, s] * model.campaignLengts[n, s] for p in model.p for n in model.n)
def auxiliary_con_rule(model, p, n, s):
    if n == model.n.last():
        return Constraint.Skip
    else:
        return model.auxiliaryVariable[p, n, s] == 0.5 * (model.productInventory[p, n+1, s] + model.productInventory[p, n, s]) - model.InventoryLowerBound[p]
def campaign_cost_per_ton_rule(model, s):
    return model.costPerTon[s] * model.cycleTime[s] * sum(model.DemandPerDay[p, s] for p in model.p) == model.setupCost[s] + model.variableCost[s]
def sequence_rule(model, p, n, s):
    if n == model.n.last():
        return Constraint.Skip
    return 1 - model.assignProductToCampaign[p, n, s] >= model.assignProductToCampaign[p, n+1, s]
def break_symmetry_rule(model, n, s):
    if n == model.n.last():
        return Constraint.Skip
    return sum(model.assignProductToCampaign[p, n, s] for p in model.p) >= sum(model.assignProductToCampaign[p, n+1, s] for p in model.p)
def campaign_lengts_bound(model, n, s):
    return (min(model.ProductionLengts_lower[p] + model.CampaignSetupTime[p] for p in model.p), max(model.ProductionLengts_upper[p] + model.CampaignSetupTime[p] for p in model.p))

def const_model():
    model = ConcreteModel()

    # Sets
    model.S=Set(initialize=[0, 1, 2])
    model.p = Set(initialize=[1,2,3])
    model.n=Set(initialize=[1,2,3])


    # Parameters
    model.VariableInvestmentCostFactor = Param(initialize=0.3271)
    model.NumDaysInYear = Param(initialize=365)
    model.MinProductionRate = Param(model.p, initialize={1: 15, 2: 15, 3: 7})
    model.MaxProductionRate = Param(model.p, initialize={1: 50, 2: 50, 3: 50})

    model.InventoryLowerBound = Param(model.p, initialize={1: 643, 2: 536, 3: 214})
    model.InventoryUpperBound = Param(model.p, initialize={1: 4018.36, 2: 3348, 3:1339.45})
    
    model.InitialInventory = Param(model.p,mutable=True)

    model.ProductionLengts_lower = Param(model.p, initialize={1: 1,2:1,3:1})
    model.ProductionLengts_upper = Param(model.p, initialize={1: 10,2:10,3:10})

    model.ProductDemand_nominal = Param(model.p, initialize={1: 4190,2:3492,3:1397})
    model.ProductDemand_stdev = Param(model.p,mutable=True)
    model.ProductDemand = Param(model.p,model.S,mutable=True)

    model.CampaignSetupTime = Param(model.p, initialize={1: 0.4, 2: 0.2, 3: 0.1})
    model.CampaignVariableCost = Param(model.p, initialize={1: 18.8304, 2: 19.2934, 3: 19.7563},mutable=True)
    model.CampaignSetupCost = Param(model.p, initialize={1: 10, 2: 20, 3: 30})

    model.prob = Param(model.S, initialize={0: 0.3, 1: 0.3, 2: 0.4})
    for p in model.p:
        model.InitialInventory[p] = 1.1*model.InventoryLowerBound[p]
        model.ProductDemand_stdev[p]=0.1*model.ProductDemand_nominal[p]

    np.random.seed(0)
    for p in model.p:
        for s in model.S: # Assign sampled values to model.ProductDemand
            model.ProductDemand[p, s] =  np.random.normal(value(model.ProductDemand_nominal[p]), value(model.ProductDemand_stdev[p]))


    model.TotalDemandPerDay = Param(model.S,mutable=True)
    placeholder={(p,s): 1/model.NumDaysInYear*model.ProductDemand[p,s] for p in model.p for s in model.S}
    model.DemandPerDay = Param(model.p, model.S,initialize=placeholder)
    # for p in model.p:
    #     for s in model.S:
    #         model.DemandPerDay[p,s] = 1/model.NumDaysInYear*model.ProductDemand[p,s]
    for s in model.S:
        model.TotalDemandPerDay[s] = sum([model.DemandPerDay[p,s] for p in model.p])

    for p in model.p:
        model.CampaignVariableCost[p] = model.CampaignVariableCost[p]/model.NumDaysInYear


    # Variables first stage
    model.productTankSize = Var(model.p, within=NonNegativeReals, initialize={1: 643, 2: 536, 3: 214})
    for p in model.p:
        model.productTankSize[p].setub(model.InventoryUpperBound[p])
        model.productTankSize[p].setlb(model.InventoryLowerBound[p])

    # Variables second stage

    model.campaignDuration = Var(model.n, model.S, within=NonNegativeReals,)
    for s in model.S:
        for n in model.n:
            model.campaignDuration[n, s].setub(max(model.ProductionLengts_upper[p] for p in model.p))
            model.campaignDuration[n, s].setlb(0)


    model.amtProductInCampaign = Var(model.p, model.n, model.S, within=NonNegativeReals)
    for p in model.p:
        for n in model.n:
            for s in model.S:
                model.amtProductInCampaign[p, n, s].setub(model.MaxProductionRate[p] * model.ProductionLengts_upper[p])
                model.amtProductInCampaign[p, n, s].setlb(0)


    model.productInventory = Var(model.p, model.n, model.S, within=NonNegativeReals)
    for p in model.p:
        for n in model.n:
            for s in model.S:
                model.productInventory[p, n, s].setub(model.InventoryUpperBound[p])
                model.productInventory[p, n, s].setlb(model.InventoryLowerBound[p])
    for s in model.S:
        model.productInventory[1, 1, s].fix(model.InventoryLowerBound[1])
    

    model.auxiliaryVariable = Var(model.p, model.n,model.S, within=NonNegativeReals)
    for p in model.p:
        for n in model.n:
            for s in model.S:
                model.auxiliaryVariable[p, n, s].setub((model.InventoryUpperBound[p] - model.InventoryLowerBound[p]) / 30)
                model.auxiliaryVariable[p, n, s].setlb(0)

    model.investmentCost = Var(model.S, within=NonNegativeReals)
    model.setupCost = Var(model.S, within=NonNegativeReals)
    model.variableCost = Var(model.S, within=NonNegativeReals)

    model.cycleTime = Var(model.S, within=NonNegativeReals)
    for s in model.S:
        model.cycleTime[s].setlb(0)
        model.cycleTime[s].setub(len(model.n) * max([model.ProductionLengts_upper[p] for p in model.p]) + sum(model.CampaignSetupTime[p] for p in model.p))

    model.costPerTon = Var(model.S, within=NonNegativeReals, bounds=(0,100))

    model.campaignLengts = Var(model.n, model.S, within=NonNegativeReals,bounds=campaign_lengts_bound)
    model.assignProductToCampaign = Var(model.p, model.n, model.S, within=Binary)


    # Constraints first stage


    # Constraints second stage
    for s in model.S:
        model.assignProductToCampaign[1, 1, s].fix(1)

    for s in model.S:
        model.assignProductToCampaign[2, 1, s].fix(0)

    for s in model.S:
        model.assignProductToCampaign[3, 1, s].fix(0)

    for s in model.S:
        model.assignProductToCampaign[1, 2, s].fix(0)

    model.cycleTime_constraint = Constraint(model.S,expr=cycleTime_constraint)

    model.UniqueProduct = Constraint(model.n, model.S, rule=unique_product_rule)

    model.MaterialBalance = Constraint(model.p, model.n, model.S, rule=material_balance_rule)

    model.TankCapacity = Constraint(model.p, model.n, model.S, rule=tank_capacity_rule)

    model.ProductionUpperBound1 = Constraint(model.p, model.n, model.S, rule=production_upper_bound1_rule)

    model.ProductionLowerBound1 = Constraint(model.p, model.n, model.S, rule=production_lower_bound1_rule)

    model.ProductionUpperBound2 = Constraint(model.p, model.n, model.S, rule=production_upper_bound2_rule)

    model.ProductionLowerBound2 = Constraint(model.p, model.n, model.S, rule=production_lower_bound2_rule)

    model.CampaignUpperBound = Constraint(model.n, model.S, rule=campaign_upper_bound_rule)

    model.CampaignLowerBound = Constraint(model.n, model.S, rule=campaign_lower_bound_rule)

    model.CampanLengtsDef = Constraint(model.n, model.S, rule=campaign_lengts_def_rule)

    model.CampaignSetupCostCon = Constraint(model.S, rule=campaign_setup_cost_con_rule)

    model.CampaignStorageCost = Constraint(model.S, rule=campaign_storage_cost_rule)

    model.AuxiliaryCon = Constraint(model.p, model.n, model.S, rule=auxiliary_con_rule)

    model.CampaignCostPerTon = Constraint(model.S, rule=campaign_cost_per_ton_rule)

    model.Sequence = Constraint(model.p, model.n, model.S, rule=sequence_rule)

    model.BreakSymmetry = Constraint(model.n, model.S, rule=break_symmetry_rule)

    # Build model
    builder = StoModelBuilder('pyomo', name='tanksize', m_type='NLP', hint=False)
    scenarios = list(model.S)
    var1_names = ['productTankSize']
    con1_names = []
    def _obj(m, s):
        return m.prob[s] * m.costPerTon[s] + m.VariableInvestmentCostFactor*m.prob[s] / m.TotalDemandPerDay[s] * sum(sqrt(m.productTankSize[p]) for p in m.p)

    _content = {
        'pm': model,
        'y_set': var1_names,
        'scenarios': scenarios,
        'con_1': con1_names,
        'objs': {s: _obj for s in scenarios},
        'obj_sense': 1
    }
    sto_m=builder.build(**_content)
    return sto_m
