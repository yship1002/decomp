###############################################################################
# Flowsheet Decomposition: Three ED Units in Series with Recycle
#
# Description:
#   This flowsheet script models three electrodialysis (ED) units in series
#   with recycle streams, broken into a decomposed formulation suitable for
#   branch-and-bound optimization. The first-stage variables are only the
#   copies of the broken connecting streams (3, 5, 6, 8). The second-stage
#   variables are the unit-local decision and outlet variables for each ED unit.
#
# Flowsheet Topology:
#   Feed (1) → ED1 → Stream 2 (dilute outlet / treated discharge)
#                    ↓ Stream 3 (concentrate) → ED2 → Stream 5 (concentrate) → ED3 → Stream 7 (concentrate product)
#                                           ↑ Stream 6 (dilute from ED2 to recycle into ED1)
#                                           ↓ Stream 8 (dilute from ED3 back to ED2)
#
# Broken Streams (First-Stage Interface Variables):
#   Stream 3: ED1 concentrate outlet → ED2 inlet (flow_S3 and conc_S3)
#   Stream 5: ED2 concentrate outlet → ED3 inlet (flow_S5 and conc_S5)
#   Stream 6: ED2 dilute outlet → ED1 recycle (flow_S6 and conc_S6)
#   Stream 8: ED3 dilute → ED2 recycle (flow_S8 and conc_S8)
#
# Second-Stage Variables (Unit-Local):
#   For each unit i (1, 2, 3):
#     - Membrane design: memLength[i], memWidth[i]
#     - Operating: I[i] (current), flowSplit[i]
#     - Outlet concentrations: concOutDilute[i], concOutConcentrate[i]
#     - Outlet flows: flowOutDilute[i], flowOutConcentrate[i]
#     - Voltage and resistance: voltCellPair[i], resConcentrate[i], resDilute[i]
#     - Cost terms: capex[i], opex1[i], opex2[i]
#
# Constraints:
#   - Unit-local ED equations: mass balance, component balance, voltage, etc.
#   - Linking constraints: connect broken-stream copies to unit outlets
#   - Recycle consistency: enforce balance on recycle loops
#
# Objective:
#   Minimize total cost across all three units.
#
###############################################################################
from ...main import StoModelBuilder
import pyomo.environ as pyo
import numpy as np

###############################################################################
# Build the Flowsheet Model
###############################################################################

def const_model(returnPyomoModel: bool = False):
    """
    Build the three-unit ED flowsheet with first-stage broken-stream
    interface variables and second-stage unit-local variables/equations.
    """
    
    pm = pyo.ConcreteModel()
    # For each unit (1, 2, 3), define a subdictionary of variables
    pm.units = pyo.Set(initialize=[1, 2, 3])

    # =========================================================================
    # PARAMETERS: Shared physical/operational parameters
    # =========================================================================
    
    pm.numCells          = pyo.Param(initialize=500)
    pm.daysOperation     = pyo.Param(initialize=7 * 365)
    pm.Rg                = pyo.Param(initialize=8.314)
    pm.temp              = pyo.Param(initialize=273.15 + 25)
    pm.faraday           = pyo.Param(initialize=96485.3321)
    pm.molweightN        = pyo.Param(initialize=14.0067)
    pm.molweightNH4Cl    = pyo.Param(initialize=53.491)
    pm.thicknessCEM      = pyo.Param(initialize=0.00011)
    pm.thicknessAEM      = pyo.Param(initialize=0.000105)
    pm.permSelCEM        = pyo.Param(initialize=0.9)
    pm.permSelAEM        = pyo.Param(initialize=0.9)
    pm.transCEM          = pyo.Param(initialize=0.95)
    pm.transAEM          = pyo.Param(initialize=0.95)
    pm.transIonConc      = pyo.Param(initialize=0.491)
    pm.transIonDil       = pyo.Param(initialize=0.509)
    pm.saltDiffCEM       = pyo.Param(initialize=1e-12)
    pm.saltDiffAEM       = pyo.Param(initialize=1e-12)
    pm.saltDiffConc      = pyo.Param(initialize=1.84e-9)
    pm.saltDiffDil       = pyo.Param(initialize=1.84e-9)
    pm.waterPermCEM      = pyo.Param(initialize=7.79e-11 / 3600)
    pm.waterPermAEM      = pyo.Param(initialize=6.29e-11 / 3600)
    pm.vantHoffNumber    = pyo.Param(initialize=2)
    pm.molweightH2O      = pyo.Param(initialize=0.018)   # kg/mol
    pm.molweightH2OgMol  = pyo.Param(initialize=18.01528)  # g/mol
    pm.densityH2O        = pyo.Param(initialize=1000)
    pm.densityH2OGM3     = pyo.Param(initialize = 1000 * 1000)   # g/m3
    pm.densityManure     = pyo.Param(initialize=1000)
    pm.viscosityManure   = pyo.Param(initialize=1e-3)
    pm.waterTransNumber  = pyo.Param(initialize=6 + 8 + 2, mutable=True)
    pm.osmoticCoeff      = pyo.Param(initialize=1)
    pm.resBlank          = pyo.Param(initialize=2.5e-5)
    pm.resCEM            = pyo.Param(initialize=2.5e-4)
    pm.resAEM            = pyo.Param(initialize=1.8e-4)
    pm.conducConcentrate = pyo.Param(initialize=149.6e-4)
    pm.conducDilute      = pyo.Param(initialize=149.6e-4)
    pm.costElec          = pyo.Param(initialize=0.08147)
    # Original (old) scaling defaults commented out
    # pm.scaleFacFlow       = pyo.Param(pm.units,mutable=True)
    # for i in pm.units:
    #     pm.scaleFacFlow[i] = 192555 / (1000 * 24 * 60 * 60)
    # pm.scaleFacConc       = pyo.Param(pm.units,initialize={1: 1366 / pm.molweightN, 2: 1366 / pm.molweightN, 3: 1366 / pm.molweightN}, mutable=True)

    # Tuned uniform scaling
    # - scaleFacFlow: ~5.388e-06 m3/s per cell
    # - scaleFacConc: ~3.967e+02 mol/m3
    pm.scaleFacFlow = pyo.Param(pm.units, mutable=True, initialize={1: 5.388e-06, 2: 5.388e-06, 3: 5.388e-06})
    pm.scaleFacConc = pyo.Param(pm.units, mutable=True, initialize={1: 3.967e+02, 2: 3.967e+02, 3: 3.967e+02})
    
    # =========================================================================
    # FIRST-STAGE VARIABLES
    # =========================================================================
    
    # stream Feed: Flowsheet inlet feed → ED1 inlet (defined as parameter since it's fixed)
    pm.flowFeed = pyo.Param(initialize=192555 / (1000 * 24 * 60 * 60), mutable=True)   # units: m3/s
    pm.concFeed = pyo.Param(initialize= 1366, mutable=True)                            # units: mg-N/L
    
    pm.molNFeed = pyo.Expression(expr = pm.flowFeed * pm.concFeed/pm.molweightN)                # units: mol/s
    pm.molWFeed = pyo.Expression(expr = pm.flowFeed * (pm.densityH2OGM3 - pm.concFeed)/pm.molweightH2OgMol)            # units: mol/s
    
    # all flowsheet level molar flows (mol x Stream x) have units of mol/s
    # stream 1: Mixer1 (feed + recycle from ED1) → ED1 inlet (defined as variables in first stage)
    pm.molNStream1 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 11), initialize=10)
    pm.molWStream1 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 500), initialize=1)

    # stream 2: ED1 concentrate outlet → concentrate discard (defined as variables in first stage)
    pm.molNStream2 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 12), initialize=10)
    pm.molWStream2 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 500), initialize=1)

    # Stream 3: ED1 dilute outlet → ED2 inlet
    pm.molNStream3 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 13), initialize=10)
    pm.molWStream3 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 500), initialize=1)

    # stream 4: Mixer2 (concentrate from ED1 + recycle from ED2) → ED2 inlet (defined as variables in first stage)
    pm.molNStream4 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 14), initialize=10)
    pm.molWStream4 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 500), initialize=1)
    
    # Stream 5: ED2 dilute outlet → ED3 inlet
    pm.molNStream5 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 15), initialize=10)
    pm.molWStream5 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 500), initialize=1)
    
    # Stream 6: ED2 concentrate → ED1 recycle
    pm.molNStream6 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 16), initialize=1)
    pm.molWStream6 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 500), initialize=1)

    # stream 7: ED3 concentrate outlet → flowsheet product (defined as variables in first stage)
    pm.molNStream7 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 17), initialize=10)
    pm.molWStream7 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 500), initialize=1)
    
    # Stream 8: ED3 dilute → ED2 recycle
    pm.molNStream8 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 18), initialize=1)
    pm.molWStream8 = pyo.Var(within=pyo.NonNegativeReals, bounds=(1e-5, 500), initialize=1)

    # =========================================================================
    # FIRST-STAGE CONSTRAINTS
    # =========================================================================

    # Mixer 1: Feed + ED1 Dilute Recycle → ED1 Inlet
    pm.conMolNMixer1 = pyo.Constraint(expr = pm.molNStream1 - pm.molNFeed - pm.molNStream6 == 0)
    pm.conMolWMixer1 = pyo.Constraint(expr = pm.molWStream1 - pm.molWFeed - pm.molWStream6 == 0)

    # Mixer 2: ED1 Concentrate + ED2 Dilute Recycle → ED2 Inlet
    pm.conMolNMixer2 = pyo.Constraint(expr = pm.molNStream4 - pm.molNStream3 - pm.molNStream8 == 0)
    pm.conMolWMixer2 = pyo.Constraint(expr = pm.molWStream4 - pm.molWStream3 - pm.molWStream8 == 0)

    # Purity targets on ED1 dilute outlet (Stream 2) and ED3 concentrate outlet (Stream 7).
    # mg-N/L definition (assuming aqueous solution with 1 L ~ 1000 g water):
    #   mgN/L = (molN * MW_N [g/mol]) / (molW * MW_H2O [g/mol]) * 1e6
    # Rearranged to a linear constraint in (molN, molW):
    #   molN <= Cmax * (MW_H2O / MW_N / 1e6) * molW
    #   molN >= Cmin * (MW_H2O / MW_N / 1e6) * molW
    pm.conPurityED1 = pyo.Constraint(
        expr=pm.molNStream2 <= 400 * (pm.molweightH2OgMol / pm.molweightN / 1e6) * pm.molWStream2
    )
    pm.conPurityED3 = pyo.Constraint(
        expr=pm.molNStream7 >= 30000 * (pm.molweightH2OgMol / pm.molweightN / 1e6) * pm.molWStream7
    )
    

    # =========================================================================
    # SECOND-STAGE VARIABLES: Unit-Local ED Model Variables for Each Unit
    # =========================================================================
    
    # Unit-local decision variables
    pm.I                    = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-1, 1e3), initialize={1:20, 2:0.1, 3:35})
    pm.flowSplit            = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 1), initialize={1:0.85, 2:0.37, 3:0})
    pm.memLength            = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0.01, 100), initialize={1:0.96, 2:0.01, 3:1.3})
    pm.memWidth             = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0.01, 10), initialize={1:0.2, 2:0.11, 3:0.1})
    pm.thicknessConcentrate = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0.001, 1), initialize=0.001)
    pm.thicknessDilute      = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0.001, 1), initialize=0.001)
    
    # Unit-local outlet variables
    pm.flowOutConcentrateND = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-3, 100), initialize=0.01)
    pm.flowOutDiluteND      = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-3, 10), initialize={1:0.99, 2:0.5, 3:0.5})
    pm.concOutConcentrateND = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-3, 100), initialize={1:20, 2:5, 3:1})
    pm.concOutDiluteND      = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-3, 100), initialize=0.3)

    # NOTE: If you want the molar-flow model to match the old volumetric model's
    # objective/solution, you likely need to re-introduce the same product/purity
    # specifications here (and ensure they apply to the correct outlet streams).
    
    # Unit-local voltage and resistance
    pm.voltCellPair    = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 5), initialize={1:0.38, 2:0.33, 3:0.63})
    pm.resConcentrate  = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-8, 1), initialize=1e-5)
    pm.resDilute       = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-8, 1), initialize=1e-5)
    pm.voltNonOhmicCEM = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-7, 1), initialize=0.01)
    pm.voltNonOhmicAEM = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-7, 1), initialize=0.01)

    
    # Unit-local cost variables
    pm.capex     = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 1e6), initialize=1e6)
    pm.opex1     = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 1e6), initialize=1e6)
    pm.opex2     = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 1e6), initialize=1e6)
    pm.costTotal = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 1e6), initialize=1e6)
    
    
    # =========================================================================
    # EXPRESSIONS: Unit-Local Inlet Conditions from Flowsheet Interface
    # =========================================================================
    
    # first, convert flowsheet level molar flows to volumetric flows and concentrations
    pm.massStream1 = pyo.Expression(expr = pm.molNStream1 * pm.molweightN + pm.molWStream1 * pm.molweightH2OgMol)   # units: g/s
    pm.flowStream1 = pyo.Expression(expr = pm.massStream1 / pm.densityH2OGM3)   # units: m3/s

    pm.concStream1 = pyo.Expression(expr = pm.molNStream1 / pm.flowStream1)   # units: mol/m3

    # Unit 1: stream 1 from mixer 1 (feed + recycle from ED1)
    pm.flowInED1 = pyo.Expression(expr = pm.flowStream1)
    pm.concInED1 = pyo.Expression(expr = pm.concStream1)

    pm.massStream4 = pyo.Expression(expr = pm.molNStream4 * pm.molweightN + pm.molWStream4 * pm.molweightH2OgMol)   # units: g/s
    pm.flowStream4 = pyo.Expression(expr = pm.massStream4 / pm.densityH2OGM3)   # units: m3/s

    pm.concStream4 = pyo.Expression(expr = pm.molNStream4 / pm.flowStream4)   # units: mol/m3

    # Unit 2: stream 4 from mixer 2 (concentrate from ED1 + recycle from ED2)
    pm.flowInED2 = pyo.Expression(expr = pm.flowStream4)
    pm.concInED2 = pyo.Expression(expr = pm.concStream4)
    
    pm.massStream5 = pyo.Expression(expr = pm.molNStream5 * pm.molweightN + pm.molWStream5 * pm.molweightH2OgMol)   # units: g/s
    pm.flowStream5In = pyo.Expression(expr = pm.massStream5 / pm.densityH2OGM3)   # units: m3/s

    pm.concStream5In = pyo.Expression(expr = pm.molNStream5 / pm.flowStream5In)   # units: mol/m3

    # Unit 3: stream 5 from ED2 dilute outlet → ED3 inlet
    pm.flowInED3 = pyo.Expression(expr = pm.flowStream5In)
    pm.concInED3 = pyo.Expression(expr = pm.concStream5In)
    
    # =========================================================================
    # UNIT MODELS: Create Three Independent ED Unit Submodels
    # ========================================================================

    def flowInConcRule(m, unitIdx):
        if unitIdx == 1:
            return m.flowSplit[unitIdx] * m.flowInED1 / m.numCells
        if unitIdx == 2:
            return m.flowSplit[unitIdx] * m.flowInED2 / m.numCells
        return m.flowSplit[unitIdx] * m.flowInED3 / m.numCells
    pm.flowInConc = pyo.Expression(pm.units, rule=flowInConcRule)

    def flowInDilRule(m, unitIdx):
        if unitIdx == 1:
            return (1 - m.flowSplit[unitIdx]) * m.flowInED1 / m.numCells
        if unitIdx == 2:
            return (1 - m.flowSplit[unitIdx]) * m.flowInED2 / m.numCells
        return (1 - m.flowSplit[unitIdx]) * m.flowInED3 / m.numCells
    pm.flowInDil = pyo.Expression(pm.units, rule=flowInDilRule)

    def concInConcRule(m, unitIdx):
        if unitIdx == 1:
            return m.concInED1
        if unitIdx == 2:
            return m.concInED2
        return m.concInED3
    pm.concInConc = pyo.Expression(pm.units, rule=concInConcRule)

    def concInDilRule(m, unitIdx):
        if unitIdx == 1:
            return m.concInED1
        if unitIdx == 2:
            return m.concInED2
        return m.concInED3
    pm.concInDil = pyo.Expression(pm.units, rule=concInDilRule)

    def uConcRule(m, unitIdx):
        return 0.5 * (m.flowOutConcentrateND[unitIdx] * m.scaleFacFlow[unitIdx] + m.flowInConc[unitIdx]) / (
            m.memWidth[unitIdx] * m.thicknessConcentrate[unitIdx]
        )
    pm.uConc = pyo.Expression(pm.units, rule=uConcRule)

    def uDilRule(m, unitIdx):
        return 0.5 * (m.flowOutDiluteND[unitIdx] * m.scaleFacFlow[unitIdx] + m.flowInDil[unitIdx]) / (
            m.memWidth[unitIdx] * m.thicknessDilute[unitIdx]
        )
    pm.uDil = pyo.Expression(pm.units, rule=uDilRule)

    def reConcRule(m, unitIdx):
        return m.densityManure * m.uConc[unitIdx] * (2 * m.thicknessConcentrate[unitIdx]) / m.viscosityManure
    pm.reConc = pyo.Expression(pm.units, rule=reConcRule)

    def reDilRule(m, unitIdx):
        return m.densityManure * m.uDil[unitIdx] * (2 * m.thicknessDilute[unitIdx]) / m.viscosityManure
    pm.reDil = pyo.Expression(pm.units, rule=reDilRule)

    def scConcRule(m, unitIdx):
        return m.viscosityManure / (m.densityManure * m.saltDiffConc)
    pm.scConc = pyo.Expression(pm.units, rule=scConcRule)

    def scDilRule(m, unitIdx):
        return m.viscosityManure / (m.densityManure * m.saltDiffDil)
    pm.scDil = pyo.Expression(pm.units, rule=scDilRule)

    def shConcCemRule(m, unitIdx):
        return 0.29 * (m.reConc[unitIdx] ** 0.5) * (m.scConc[unitIdx] ** 0.33)
    pm.shConcCEM = pyo.Expression(pm.units, rule=shConcCemRule)

    def shDilCemRule(m, unitIdx):
        return 0.29 * (m.reDil[unitIdx] ** 0.5) * (m.scDil[unitIdx] ** 0.33)
    pm.shDilCEM = pyo.Expression(pm.units, rule=shDilCemRule)

    def shConcAemRule(m, unitIdx):
        return 0.29 * (m.reConc[unitIdx] ** 0.5) * (m.scConc[unitIdx] ** 0.33)
    pm.shConcAEM = pyo.Expression(pm.units, rule=shConcAemRule)

    def shDilAemRule(m, unitIdx):
        return 0.29 * (m.reDil[unitIdx] ** 0.5) * (m.scDil[unitIdx] ** 0.33)
    pm.shDilAEM = pyo.Expression(pm.units, rule=shDilAemRule)

    def avgConcConcRule(m, unitIdx):
        return 0.5 * (m.concOutConcentrateND[unitIdx] * m.scaleFacConc[unitIdx] + m.concInConc[unitIdx])
    pm.avgConcConc = pyo.Expression(pm.units, rule=avgConcConcRule)

    def avgConcDilRule(m, unitIdx):
        return 0.5 * (m.concOutDiluteND[unitIdx] * m.scaleFacConc[unitIdx] + m.concInDil[unitIdx])
    pm.avgConcDil = pyo.Expression(pm.units, rule=avgConcDilRule)

    def iLimConcCEMRule(m, unitIdx):
        return m.shConcCEM[unitIdx] * m.avgConcConc[unitIdx] * m.faraday * m.saltDiffConc / (
            2 * m.thicknessConcentrate[unitIdx] * (m.transCEM - m.transIonConc)
        )
    pm.iLimConcCEM = pyo.Expression(pm.units, rule=iLimConcCEMRule)

    def iLimDilCEMRule(m, unitIdx):
        return m.shDilCEM[unitIdx] * m.avgConcDil[unitIdx] * m.faraday * m.saltDiffDil / (
            2 * m.thicknessDilute[unitIdx] * (m.transCEM - m.transIonConc)
        )
    pm.iLimDilCEM = pyo.Expression(pm.units, rule=iLimDilCEMRule)

    def iLimConcAEMRule(m, unitIdx):
        return m.shConcAEM[unitIdx] * m.avgConcConc[unitIdx] * m.faraday * m.saltDiffConc / (
            2 * m.thicknessConcentrate[unitIdx] * (m.transAEM - m.transIonDil)
        )
    pm.iLimConcAEM = pyo.Expression(pm.units, rule=iLimConcAEMRule)

    def iLimDilAEMRule(m, unitIdx):
        return m.shDilAEM[unitIdx] * m.avgConcDil[unitIdx] * m.faraday * m.saltDiffDil / (
            2 * m.thicknessDilute[unitIdx] * (m.transAEM - m.transIonDil)
        )
    pm.iLimDilAEM = pyo.Expression(pm.units, rule=iLimDilAEMRule)

    def concInConcIntCEMRule(m, unitIdx):
        return m.concInConc[unitIdx] * (1 + m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx]) / m.iLimConcCEM[unitIdx])
    pm.concInConcIntCEM = pyo.Expression(pm.units, rule=concInConcIntCEMRule)

    def concInDilIntCEMRule(m, unitIdx):
        return m.concInDil[unitIdx] * (1 - m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx]) / m.iLimDilCEM[unitIdx])
    pm.concInDilIntCEM = pyo.Expression(pm.units, rule=concInDilIntCEMRule)

    def concInConcIntAEMRule(m, unitIdx):
        return m.concInConc[unitIdx] * (1 + m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx]) / m.iLimConcAEM[unitIdx])
    pm.concInConcIntAEM = pyo.Expression(pm.units, rule=concInConcIntAEMRule)

    def concInDilIntAEMRule(m, unitIdx):
        return m.concInDil[unitIdx] * (1 - m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx]) / m.iLimDilAEM[unitIdx])
    pm.concInDilIntAEM = pyo.Expression(pm.units, rule=concInDilIntAEMRule)

    def concOutConcIntCEMRule(m, unitIdx):
        return m.concOutConcentrateND[unitIdx] * (1 + m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx]) / m.iLimConcCEM[unitIdx])
    pm.concOutConcIntCEM = pyo.Expression(pm.units, rule=concOutConcIntCEMRule)

    def concOutDilIntCEMRule(m, unitIdx):
        return m.concOutDiluteND[unitIdx] * (1 - m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx]) / m.iLimDilCEM[unitIdx])
    pm.concOutDilIntCEM = pyo.Expression(pm.units, rule=concOutDilIntCEMRule)

    def concOutConcIntAEMRule(m, unitIdx):
        return m.concOutConcentrateND[unitIdx] * (1 + m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx]) / m.iLimConcAEM[unitIdx])
    pm.concOutConcIntAEM = pyo.Expression(pm.units, rule=concOutConcIntAEMRule)

    def concOutDilIntAEMRule(m, unitIdx):
        return m.concOutDiluteND[unitIdx] * (1 - m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx]) / m.iLimDilAEM[unitIdx])
    pm.concOutDilIntAEM = pyo.Expression(pm.units, rule=concOutDilIntAEMRule)

    def avgConcConcIntCEMRule(m, unitIdx):
        return 0.5 * (m.concOutConcIntCEM[unitIdx] * m.scaleFacConc[unitIdx] + m.concInConcIntCEM[unitIdx])
    pm.avgConcConcIntCEM = pyo.Expression(pm.units, rule=avgConcConcIntCEMRule)

    def avgConcDilIntCEMRule(m, unitIdx):
        return 0.5 * (m.concOutDilIntCEM[unitIdx] * m.scaleFacConc[unitIdx] + m.concInDilIntCEM[unitIdx])
    pm.avgConcDilIntCEM = pyo.Expression(pm.units, rule=avgConcDilIntCEMRule)

    def avgConcConcIntAEMRule(m, unitIdx):
        return 0.5 * (m.concOutConcIntAEM[unitIdx] * m.scaleFacConc[unitIdx] + m.concInConcIntAEM[unitIdx])
    pm.avgConcConcIntAEM = pyo.Expression(pm.units, rule=avgConcConcIntAEMRule)

    def avgConcDilIntAEMRule(m, unitIdx):
        return 0.5 * (m.concOutDilIntAEM[unitIdx] * m.scaleFacConc[unitIdx] + m.concInDilIntAEM[unitIdx])
    pm.avgConcDilIntAEM = pyo.Expression(pm.units, rule=avgConcDilIntAEMRule)

    def condFluxRule(m, unitIdx):
        return (m.transCEM - (1 - m.transAEM)) * (m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx] * m.faraday))
    pm.condFlux = pyo.Expression(pm.units, rule=condFluxRule)

    def diffFluxAEMRule(m, unitIdx):
        return -(m.saltDiffAEM / m.thicknessAEM) * (m.avgConcConcIntAEM[unitIdx] - m.avgConcDilIntAEM[unitIdx])
    pm.diffFluxAEM = pyo.Expression(pm.units, rule=diffFluxAEMRule)

    def diffFluxCEMRule(m, unitIdx):
        return -(m.saltDiffCEM / m.thicknessCEM) * (m.avgConcConcIntCEM[unitIdx] - m.avgConcDilIntCEM[unitIdx])
    pm.diffFluxCEM = pyo.Expression(pm.units, rule=diffFluxCEMRule)

    def fluxIonsRule(m, unitIdx):
        return m.condFlux[unitIdx] + m.diffFluxAEM[unitIdx] + m.diffFluxCEM[unitIdx]
    pm.fluxIonsTotal = pyo.Expression(pm.units, rule=fluxIonsRule)

    def osmWaterFluxAEMRule(m, unitIdx):
        return m.waterPermAEM * m.vantHoffNumber * m.Rg * m.temp * (m.osmoticCoeff * m.avgConcConcIntAEM[unitIdx] - m.osmoticCoeff * m.avgConcDilIntAEM[unitIdx])
    pm.osmWaterFluxAEM = pyo.Expression(pm.units, rule=osmWaterFluxAEMRule)

    def osmWaterFluxCEMRule(m, unitIdx):
        return m.waterPermCEM * m.vantHoffNumber * m.Rg * m.temp * (m.osmoticCoeff * m.avgConcConcIntCEM[unitIdx] - m.osmoticCoeff * m.avgConcDilIntCEM[unitIdx])
    pm.osmWaterFluxCEM = pyo.Expression(pm.units, rule=osmWaterFluxCEMRule)

    def eosmWaterFluxRule(m, unitIdx):
        return m.waterTransNumber * m.fluxIonsTotal[unitIdx] * m.molweightH2O / m.densityH2O
    pm.eosmWaterFlux = pyo.Expression(pm.units, rule=eosmWaterFluxRule)

    def fluxWaterTotalRule(m, unitIdx):
        return m.osmWaterFluxAEM[unitIdx] + m.osmWaterFluxCEM[unitIdx] + m.eosmWaterFlux[unitIdx]
    pm.fluxWaterTotal = pyo.Expression(pm.units, rule=fluxWaterTotalRule)



    ######################################################################################################
    # CONSTRUCT CONSTRAINTS
    ######################################################################################################

    def resConcRule(m, unitIdx):
        return m.resConcentrate[unitIdx] * m.conducConcentrate * m.avgConcConc[unitIdx] - m.thicknessConcentrate[unitIdx] == 0
    pm.resConcConstraint = pyo.Constraint(pm.units, rule=resConcRule)

    def resDilRule(m, unitIdx):
        return m.resDilute[unitIdx] * m.conducDilute * m.avgConcDil[unitIdx] - m.thicknessDilute[unitIdx] == 0
    pm.resDilConstraint = pyo.Constraint(pm.units, rule=resDilRule)

    def voltNonOhmicCEMRule(m, unitIdx):
        return pyo.exp(m.voltNonOhmicCEM[unitIdx] * m.faraday / (m.permSelCEM * m.Rg * m.temp)) * m.avgConcDilIntCEM[unitIdx] - m.avgConcConcIntCEM[unitIdx] == 0
    pm.voltNonOhmicCEMConstraint = pyo.Constraint(pm.units, rule=voltNonOhmicCEMRule)

    def voltNonOhmicAEMRule(m, unitIdx):
        return pyo.exp(m.voltNonOhmicAEM[unitIdx] * m.faraday / (m.permSelAEM * m.Rg * m.temp)) * m.avgConcDilIntAEM[unitIdx] - m.avgConcConcIntAEM[unitIdx] == 0
    pm.voltNonOhmicAEMConstraint = pyo.Constraint(pm.units, rule=voltNonOhmicAEMRule)

    def resOhmicRule(m, unitIdx):
        return m.resConcentrate[unitIdx] + m.resDilute[unitIdx] + m.resCEM + m.resAEM
    pm.resOhmic = pyo.Expression(pm.units, rule=resOhmicRule)

    def voltNonOhmRule(m, unitIdx):
        return m.voltNonOhmicCEM[unitIdx] + m.voltNonOhmicAEM[unitIdx]
    pm.voltNonOhm = pyo.Expression(pm.units, rule=voltNonOhmRule)

    def massBalConcentrateRule(m, unitIdx):
        return m.flowOutConcentrateND[unitIdx] - (m.flowInConc[unitIdx] / m.scaleFacFlow[unitIdx]) - \
               (m.fluxWaterTotal[unitIdx] * m.memLength[unitIdx] * m.memWidth[unitIdx] / m.scaleFacFlow[unitIdx]) - \
               ((m.molweightNH4Cl * 0.001 / m.densityH2O) * m.fluxIonsTotal[unitIdx] * \
                m.memLength[unitIdx] * m.memWidth[unitIdx] / m.scaleFacFlow[unitIdx]) == 0
    pm.conMassBalConc = pyo.Constraint(pm.units, rule=massBalConcentrateRule)

    def massBalDiluteRule(m, unitIdx):
        return m.flowOutDiluteND[unitIdx] - (m.flowInDil[unitIdx] / m.scaleFacFlow[unitIdx]) + \
               (m.fluxWaterTotal[unitIdx] * m.memLength[unitIdx] * m.memWidth[unitIdx] / m.scaleFacFlow[unitIdx]) + \
               ((m.molweightNH4Cl * 0.001 / m.densityH2O) * m.fluxIonsTotal[unitIdx] * \
                m.memLength[unitIdx] * m.memWidth[unitIdx] / m.scaleFacFlow[unitIdx]) == 0
    pm.conMassBalDil = pyo.Constraint(pm.units, rule=massBalDiluteRule)

    def compBalConcentrateRule(m, unitIdx):
        return (m.flowOutConcentrateND[unitIdx] * m.concOutConcentrateND[unitIdx]) - \
               (m.flowInConc[unitIdx] * m.concInConc[unitIdx]) / (m.scaleFacFlow[unitIdx] * m.scaleFacConc[unitIdx]) - \
               m.fluxIonsTotal[unitIdx] * m.memLength[unitIdx] * m.memWidth[unitIdx] / (m.scaleFacFlow[unitIdx] * m.scaleFacConc[unitIdx]) == 0
    pm.conCompBalConc = pyo.Constraint(pm.units, rule=compBalConcentrateRule)

    def compBalDiluteRule(m, unitIdx):
        return (m.flowOutDiluteND[unitIdx] * m.concOutDiluteND[unitIdx]) - \
               (m.flowInDil[unitIdx] * m.concInDil[unitIdx]) / (m.scaleFacFlow[unitIdx] * m.scaleFacConc[unitIdx]) + \
               m.fluxIonsTotal[unitIdx] * m.memLength[unitIdx] * m.memWidth[unitIdx] / (m.scaleFacFlow[unitIdx] * m.scaleFacConc[unitIdx]) == 0
    pm.conCompBalDil = pyo.Constraint(pm.units, rule=compBalDiluteRule)

    def voltageRule(m, unitIdx):
        return m.voltCellPair[unitIdx] - m.voltNonOhm[unitIdx] - m.resOhmic[unitIdx] * (m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx])) == 0
    pm.conVoltage = pyo.Constraint(pm.units, rule=voltageRule)

    def limitCurrentCEMRule(m, unitIdx):
        return m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx] * m.iLimDilCEM[unitIdx]) - 1 <= 0
    pm.conLimitCurrCEM = pyo.Constraint(pm.units, rule=limitCurrentCEMRule)

    def limitCurrentAEMRule(m, unitIdx):
        return m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx] * m.iLimDilAEM[unitIdx]) - 1 <= 0
    pm.conLimitCurrAEM = pyo.Constraint(pm.units, rule=limitCurrentAEMRule)

    def capexRule(m, unitIdx):
        return m.capex[unitIdx] == 6800 * m.memLength[unitIdx] * m.memWidth[unitIdx] + 2 * 100 * (m.numCells - 2) * m.memLength[unitIdx] * m.memWidth[unitIdx]
    pm.conCapex = pyo.Constraint(pm.units, rule=capexRule)

    def opex1Rule(m, unitIdx):
        voltBlank = m.resBlank * m.I[unitIdx] / (m.memLength[unitIdx] * m.memWidth[unitIdx])
        voltTotal = voltBlank + m.numCells * m.voltCellPair[unitIdx]
        return m.opex1[unitIdx] == m.costElec * (24 * 10**-3) * voltTotal * m.I[unitIdx] * m.daysOperation
    pm.conOpex1 = pyo.Constraint(pm.units, rule=opex1Rule)

    def opex2Rule(m, unitIdx):
        flowOutConc = m.flowOutConcentrateND[unitIdx] * m.scaleFacFlow[unitIdx]
        flowOutDil = m.flowOutDiluteND[unitIdx] * m.scaleFacFlow[unitIdx]
        fConc = 1400 / m.reConc[unitIdx]
        fDil = 104.5 / (m.reDil[unitIdx]**0.37)
        presDrop1 = m.densityManure * fConc * m.memLength[unitIdx] * (flowOutConc / m.memWidth[unitIdx])**2 / (4 * m.thicknessConcentrate[unitIdx])
        presDrop2 = m.densityManure * fDil * m.memLength[unitIdx] * (flowOutDil / m.memWidth[unitIdx])**2 / (4 * m.thicknessDilute[unitIdx])
        pumpingCost = m.costElec * (24 * 10**-3) * (presDrop1 * ((flowOutConc + m.flowInConc[unitIdx]) / 2) + presDrop2 * ((flowOutDil + m.flowInDil[unitIdx]) / 2)) * m.numCells * m.daysOperation
        return m.opex2[unitIdx] == pumpingCost
    pm.conopex2 = pyo.Constraint(pm.units, rule=opex2Rule)


    # convert outlet flows and concentrations to molar flows for flowsheet-level linking

    # unit 1 outlets: stream 2 (dilute), stream 3 (concentrate)
    pm.flowStream2 = pyo.Expression(expr = pm.flowOutDiluteND[1] * pm.scaleFacFlow[1]  * pm.numCells)   # units: m3/s 
    pm.concStream2 = pyo.Expression(expr = pm.concOutDiluteND[1] * pm.scaleFacConc[1])   # units: mol/m3
    def conMolNStream2Rule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molNStream2 == m.concStream2 * m.flowStream2
    pm.conMolNStream2 = pyo.Constraint(pm.units, rule=conMolNStream2Rule)   # units: mol/s 
    def conMolWStream2Rule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molWStream2 == m.flowStream2 * (pm.densityH2OGM3 - (pm.concStream2 * pm.molweightN)) / pm.molweightH2OgMol
    pm.conMolWStream2 = pyo.Constraint(pm.units, rule=conMolWStream2Rule)   # units: mol/s

    pm.flowStream3 = pyo.Expression(expr = pm.flowOutConcentrateND[1] * pm.scaleFacFlow[1] * pm.numCells)
    pm.concStream3 = pyo.Expression(expr = pm.concOutConcentrateND[1] * pm.scaleFacConc[1])   # units: mol/m3
    def conMolNStream3Rule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molNStream3 == m.concStream3 * m.flowStream3
    pm.conMolNStream3 = pyo.Constraint(pm.units, rule=conMolNStream3Rule)   # units: mol/s
    def conMolWStream3Rule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molWStream3 == m.flowStream3 * (pm.densityH2OGM3 - (pm.concStream3 * pm.molweightN)) / pm.molweightH2OgMol
    pm.conMolWStream3 = pyo.Constraint(pm.units, rule=conMolWStream3Rule)   # units: mol/s

    # unit 2 outlets: stream 5 (concentrate), stream 6 (dilute) 
    pm.flowStream5 = pyo.Expression(expr = pm.flowOutConcentrateND[2] * pm.scaleFacFlow[2] * pm.numCells)
    pm.concStream5 = pyo.Expression(expr = pm.concOutConcentrateND[2] * pm.scaleFacConc[2])
    def conMolNStream5Rule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molNStream5 == m.concStream5 * m.flowStream5
    pm.conMolNStream5 = pyo.Constraint(pm.units, rule=conMolNStream5Rule)   # units: mol/s
    def conMolWStream5Rule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molWStream5 == m.flowStream5 * (pm.densityH2OGM3 - (pm.concStream5 * pm.molweightN)) / pm.molweightH2OgMol
    pm.conMolWStream5 = pyo.Constraint(pm.units, rule=conMolWStream5Rule)   # units: mol/s

    pm.flowStream6 = pyo.Expression(expr = pm.flowOutDiluteND[2] * pm.scaleFacFlow[2] * pm.numCells)
    pm.concStream6 = pyo.Expression(expr = pm.concOutDiluteND[2] * pm.scaleFacConc[2])
    def conMolNStream6Rule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molNStream6 == m.concStream6 * m.flowStream6
    pm.conMolNStream6 = pyo.Constraint(pm.units, rule=conMolNStream6Rule)   # units: mol/s
    def conMolWStream6Rule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molWStream6 == m.flowStream6 * (pm.densityH2OGM3 - (pm.concStream6 * pm.molweightN)) / pm.molweightH2OgMol
    pm.conMolWStream6 = pyo.Constraint(pm.units, rule=conMolWStream6Rule)   # units: mol/s

    # unit 3 outlets: stream 7 (concentrate), stream 8 (dilute)
    pm.flowStream7 = pyo.Expression(expr = pm.flowOutConcentrateND[3] * pm.scaleFacFlow[3] * pm.numCells)
    pm.concStream7 = pyo.Expression(expr = pm.concOutConcentrateND[3] * pm.scaleFacConc[3])
    def conMolNStream7Rule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molNStream7 == m.concStream7 * m.flowStream7
    pm.conMolNStream7 = pyo.Constraint(pm.units, rule=conMolNStream7Rule)   # units: mol/s
    def conMolWStream7Rule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molWStream7 == m.flowStream7 * (pm.densityH2OGM3 - (pm.concStream7 * pm.molweightN)) / pm.molweightH2OgMol
    pm.conMolWStream7 = pyo.Constraint(pm.units, rule=conMolWStream7Rule)   # units: mol/s

    pm.flowStream8 = pyo.Expression(expr = pm.flowOutDiluteND[3] * pm.scaleFacFlow[3] * pm.numCells )
    pm.concStream8 = pyo.Expression(expr = pm.concOutDiluteND[3] * pm.scaleFacConc[3])      
    def conMolNStream8Rule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molNStream8 == m.concStream8 * m.flowStream8
    pm.conMolNStream8 = pyo.Constraint(pm.units, rule=conMolNStream8Rule)   # units: mol/s
    def conMolWStream8Rule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molWStream8 == m.flowStream8 * (pm.densityH2OGM3 - (pm.concStream8 * pm.molweightN)) / pm.molweightH2OgMol
    pm.conMolWStream8 = pyo.Constraint(pm.units, rule=conMolWStream8Rule)   # units: mol/s

    if returnPyomoModel:
        return pm

    # =========================================================================
    # OBJECTIVE: Minimize total cost across all units
    # =========================================================================
    builder = StoModelBuilder('pyomo', name='EDUnit', m_type='NLP', hint=False)
    scenarios = list(pm.units)
    var1_names = ["molNStream1","molWStream1","molNStream2","molWStream2","molNStream3","molWStream3",
                  "molNStream4","molWStream4","molNStream5","molWStream5","molNStream6","molWStream6","molNStream7","molWStream7","molNStream8","molWStream8"]
    con1_names = ["conMolNMixer1","conMolWMixer1","conMolNMixer2","conMolWMixer2",
                  "conPurityED1","conPurityED3"]


    def _obj(m, unitIdx):
        # tonsNTotal = (m.concOutConcentrateND[3] * m.scaleFacConc[3] * m.flowOutConcentrateND[3] * m.scaleFacFlow[3] * m.numCells * m.molweightN * (10**-6) * 7 * 365 * 86400)
        return (m.capex[unitIdx] + m.opex1[unitIdx] + m.opex2[unitIdx]) #/ tonsNTotal

    
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