###############################################################################
# Flowsheet Decomposition: Five NF Units in a Li/Co Enrich-Strip Cascade
#
# Description:
#   This flowsheet script models the five-unit nanofiltration (NF) cascade from
#   5NF_Flowsheet.py, broken into a decomposed formulation suitable for
#   branch-and-bound optimization, in the same style as flowsheet3EDMolFlowV2.py.
#   The first-stage variables are the molar-flow copies (Li, Co, Water) of every
#   broken/connecting stream in the flowsheet. The second-stage variables are the
#   unit-local decision and outlet variables for each NF unit (ported unchanged
#   from mainNFMolFlow.py / mainNF.py — no physics equations are altered here).
#
#   Both units and streams are indexed purely numerically (no string keys
#   anywhere) because pm.units and all indexed components must stay numeric.
#   The legends below are the single source of truth for what each number means.
#
# -----------------------------------------------------------------------------
# UNIT INDEX LEGEND (pm.units = {1, 2, 3, 4, 5})
# -----------------------------------------------------------------------------
#   1 = NFF   (feed stage)
#   2 = NFE1  (enriching stage 1)
#   3 = NFE2  (enriching stage 2 -- Li product unit)
#   4 = NFS1  (stripping stage 1)
#   5 = NFS2  (stripping stage 2 -- Co product unit)
#
# -----------------------------------------------------------------------------
# STREAM INDEX LEGEND (15 broken streams, each with molLiStreamN/molCoStreamN/
# molWStreamN, N = 1..15)
# -----------------------------------------------------------------------------
#    1 = NFF inlet     (Mixer 1 output: fresh feed + stream 5 + stream 12)
#    2 = NFF retentate  -> feeds Mixer 2 (stream 10, NFS1 inlet)
#    3 = NFF permeate   -> feeds Mixer 3 (stream 4, NFE1 inlet)
#    4 = NFE1 inlet    (Mixer 3 output: stream 3 + stream 8)
#    5 = NFE1 retentate -> recycle to Mixer 1 (stream 1, NFF inlet)
#    6 = NFE1 permeate  -> feeds NFE2 inlet directly (stream 7)
#    7 = NFE2 inlet    (direct passthrough of stream 6)
#    8 = NFE2 retentate -> recycle to Mixer 3 (stream 4, NFE1 inlet)
#    9 = NFE2 permeate  -> Li PRODUCT
#   10 = NFS1 inlet    (Mixer 2 output: stream 2 + stream 15)
#   11 = NFS1 retentate -> feeds NFS2 inlet directly (stream 13)
#   12 = NFS1 permeate  -> recycle to Mixer 1 (stream 1, NFF inlet)
#   13 = NFS2 inlet    (direct passthrough of stream 11)
#   14 = NFS2 retentate -> Co PRODUCT
#   15 = NFS2 permeate  -> recycle to Mixer 2 (stream 10, NFS1 inlet)
#
# Flowsheet Topology (mirrors 5NF_Flowsheet.py):
#   Fresh feed --> Mixer1(s1) --> NFF(1) --s2(Ret)--> Mixer2(s10) --> NFS1(4) --s11(Ret)--> NFS2(5, direct, s13) --s14(Ret)--> Co PRODUCT
#                                     |                                  ^                                          |
#                                     s3(Perm)                           s15(Perm, recycle)  <-----------------------
#                                     v
#                                 Mixer3(s4) --> NFE1(2) --s6(Perm)--> NFE2(3, direct, s7) --s9(Perm)--> Li PRODUCT
#                                     ^                |                        |
#                                     |                s5(Ret, recycle)         s8(Ret, recycle)
#                                     -------------------------------------------
#
#   Mixer 1 (stream 1,  NFF inlet):  s1  = freshFeed + s5 + s12
#   Mixer 3 (stream 4,  NFE1 inlet): s4  = s3 + s8
#   Mixer 2 (stream 10, NFS1 inlet): s10 = s2 + s15
#   NFE2 (no mixer, direct):         s7  = s6
#   NFS2 (no mixer, direct):         s13 = s11
#
# Second-Stage Variables (Unit-Local, indexed over pm.units = {1..5}):
#   For each unit u:
#     - Geometry: memLength[u], memWidth[u] (fixed), channelHeight[u]
#     - Operating: dP[u]
#     - Outlet ND vars: flowOutRetND[u], flowOutPermND[u], cLiRetND[u], cCoRetND[u],
#       cLiPermND[u], cCoPermND[u], Jw[u], fLi[u], fCo[u]
#     - Cost terms: capex[u], opex1[u], opex2[u]
#
# Constraints:
#   - Unit-local NF equations (unchanged physics): volume/component balances,
#     solvent flux, Spiegler-Kedem transport, limiting flux, capex/opex.
#   - Mixer / passthrough constraints: define each unit's inlet broken stream from
#     the streams feeding it (feed Param and/or upstream units' broken streams).
#   - Linking constraints: connect each unit's Ret/Perm broken-stream copies (see
#     legend above) to that unit's own outlet expressions (skip-per-unit pattern,
#     as in the ED file), written out per stream (no dynamic lookups) so every
#     comparison is a plain int == int check.
#   - Product purity / recovery targets (Li from stream 9 = NFE2 permeate, Co
#     from stream 14 = NFS2 retentate) written directly in terms of first-stage
#     broken-stream molar flows — since these are already molar quantities,
#     purity and recovery become simple linear ratios (no bilinear conc*flow
#     terms needed), unlike the flowsheet-level Block version in 5NF_Flowsheet.py.
#
# Objective:
#   Minimize total cost (capex + opex) per unit. As in flowsheet3EDMolFlowV2.py,
#   normalizing by a single unit's production (e.g. unit 3's liRecoveredTonYr)
#   is deliberately NOT done in the per-unit objective: that would reference a
#   variable from one subproblem (unit 3) inside every other subproblem's
#   objective, which breaks decomposition (same issue as ED's tonsNTotal, see
#   the commented-out normalization in flowsheet3EDMolFlowV2.py's _obj).
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
    Build the five-unit NF cascade flowsheet with first-stage broken-stream
    interface variables (molar basis) and second-stage unit-local
    variables/equations (ported unchanged from mainNFMolFlow.py). Units and
    streams are numbered per the legend at the top of this file.
    """

    pm = pyo.ConcreteModel()
    # Units: 1=NFF, 2=NFE1, 3=NFE2, 4=NFS1, 5=NFS2 (see legend above)
    pm.units = pyo.Set(initialize=[1, 2, 3, 4, 5])

    # =========================================================================
    # PARAMETERS: Shared physical/operational parameters
    # =========================================================================

    pm.Rg                 = pyo.Param(initialize=8.314, mutable=True)
    pm.temp                = pyo.Param(initialize=273.15 + 25, mutable=True)
    pm.densityH2O          = pyo.Param(initialize=997.0, mutable=True)                     # kg/m3
    pm.viscosity           = pyo.Param(initialize=1e-3, mutable=True)                      # feed viscosity [Pa.s]
    pm.molweightH2O        = pyo.Param(initialize=0.018, mutable=True)                     # kg/mol
    pm.molweightH2OgMol    = pyo.Param(initialize=18.01528, mutable=True)                  # g/mol
    pm.molweightLi         = pyo.Param(initialize=6.941, mutable=True)                     # g/mol
    pm.molweightCo         = pyo.Param(initialize=58.933, mutable=True)                    # g/mol
    pm.densityH2OGM3       = pyo.Param(initialize=997.0 * 1000)                            # g/m3

    # Membrane transport parameters (NF270-like for Li/Co)
    pm.Lp                  = pyo.Param(initialize=1.0e-11, mutable=True)                   # pure water permeance [m3/(m2.s.Pa)]
    pm.sigmaLi              = pyo.Param(initialize=0.05, mutable=True)                      # reflection coeff Li [-]
    pm.sigmaCo              = pyo.Param(initialize=0.85, mutable=True)                      # reflection coeff Co [-]
    pm.pLi                  = pyo.Param(initialize=5.0e-6, mutable=True)                    # solute permeability Li [m/s]
    pm.pCo                  = pyo.Param(initialize=2.0e-7, mutable=True)                    # solute permeability Co [m/s]
    pm.nuLi                 = pyo.Param(initialize=2.0, mutable=True)                       # van't Hoff factor LiCl
    pm.nuCo                 = pyo.Param(initialize=3.0, mutable=True)                       # van't Hoff factor CoCl2

    # Diffusivities in solution
    pm.diffLi                = pyo.Param(initialize=1.03e-9, mutable=True)                  # Li diffusivity [m2/s]
    pm.diffCo                = pyo.Param(initialize=0.72e-9, mutable=True)                  # Co diffusivity [m2/s]

    # Cost parameters
    pm.costElec              = pyo.Param(initialize=0.08, mutable=True)                     # electricity cost [$/kWh]
    pm.costMembrane          = pyo.Param(initialize=50.0, mutable=True)                     # membrane cost [$/m2/yr]
    pm.pumpEff               = pyo.Param(initialize=0.80, mutable=True)                     # pump efficiency
    pm.daysOperation         = pyo.Param(initialize=350, mutable=True)                      # operating days per year
    pm.memLifetime           = pyo.Param(initialize=5.0, mutable=True)                      # membrane lifetime [yr]
    pm.cpMax                 = pyo.Param(initialize=2.0, mutable=True)                      # max allowed CP ratio

    # Scale factors
    NUM_ELEMENTS = 5000            # number of parallel membrane elements per unit
    SCALE_FAC_FLOW = 1.0 / 5000.0  # flow scale per element [m3/s]
    SCALE_FAC_CONC = 50.0          # concentration scale [mol/m3]

    # =========================================================================
    # FIRST-STAGE VARIABLES: molar-flow copies of every broken/connecting stream
    # =========================================================================

    # Fresh feed to the flowsheet
    pm.feedFlow    = pyo.Param(initialize=1, mutable=True)      # m3/s
    pm.feedConcLi  = pyo.Param(initialize=50, mutable=True)     # mol/m3
    pm.feedConcCo  = pyo.Param(initialize=100, mutable=True)    # mol/m3

    pm.molLiFeed = pyo.Expression(expr=pm.feedFlow * pm.feedConcLi)   # mol/s
    pm.molCoFeed = pyo.Expression(expr=pm.feedFlow * pm.feedConcCo)   # mol/s
    pm.molWFeed  = pyo.Expression(expr=pm.feedFlow * (pm.densityH2OGM3 - pm.feedConcLi * pm.molweightLi
                                                        - pm.feedConcCo * pm.molweightCo) / pm.molweightH2OgMol)  # mol/s


    _liBounds = (1e-6, 1e6)
    _coBounds = (1e-6, 1e6)
    _wBounds  = (1000.0, 5e6)   
    _liInit, _coInit, _wInit = 50.0, 100.0, 55000.0

    pm.molLiStream1  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream1  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream1   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream2  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream2  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream2   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream3  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream3  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream3   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream4  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream4  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream4   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream5  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream5  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream5   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream6  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream6  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream6   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream7  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream7  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream7   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream8  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream8  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream8   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    # Stream 9 = Li PRODUCT (NFE2 permeate)
    pm.molLiStream9  = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream9  = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream9   = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream10 = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream10 = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream10  = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream11 = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream11 = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream11  = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream12 = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream12 = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream12  = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream13 = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream13 = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream13  = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    # Stream 14 = Co PRODUCT (NFS2 retentate)
    pm.molLiStream14 = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream14 = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream14  = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    pm.molLiStream15 = pyo.Var(within=pyo.NonNegativeReals, bounds=_liBounds, initialize=_liInit)
    pm.molCoStream15 = pyo.Var(within=pyo.NonNegativeReals, bounds=_coBounds, initialize=_coInit)
    pm.molWStream15  = pyo.Var(within=pyo.NonNegativeReals, bounds=_wBounds,  initialize=_wInit)

    # =========================================================================
    # FIRST-STAGE CONSTRAINTS: Mixers, passthroughs, purity/recovery targets
    # =========================================================================

    # Mixer 1 (stream 1, NFF inlet): fresh feed + stream 5 (NFE1 Ret) + stream 12 (NFS1 Perm)
    pm.conMixer1Li = pyo.Constraint(expr=pm.molLiStream1 - pm.molLiFeed - pm.molLiStream5 - pm.molLiStream12 == 0)
    pm.conMixer1Co = pyo.Constraint(expr=pm.molCoStream1 - pm.molCoFeed - pm.molCoStream5 - pm.molCoStream12 == 0)
    pm.conMixer1W  = pyo.Constraint(expr=pm.molWStream1  - pm.molWFeed  - pm.molWStream5  - pm.molWStream12  == 0)

    # Mixer 3 (stream 4, NFE1 inlet): stream 3 (NFF Perm) + stream 8 (NFE2 Ret)
    pm.conMixer4Li = pyo.Constraint(expr=pm.molLiStream4 - pm.molLiStream3 - pm.molLiStream8 == 0)
    pm.conMixer4Co = pyo.Constraint(expr=pm.molCoStream4 - pm.molCoStream3 - pm.molCoStream8 == 0)
    pm.conMixer4W  = pyo.Constraint(expr=pm.molWStream4  - pm.molWStream3  - pm.molWStream8  == 0)

    # Mixer 2 (stream 10, NFS1 inlet): stream 2 (NFF Ret) + stream 15 (NFS2 Perm)
    pm.conMixer10Li = pyo.Constraint(expr=pm.molLiStream10 - pm.molLiStream2 - pm.molLiStream15 == 0)
    pm.conMixer10Co = pyo.Constraint(expr=pm.molCoStream10 - pm.molCoStream2 - pm.molCoStream15 == 0)
    pm.conMixer10W  = pyo.Constraint(expr=pm.molWStream10  - pm.molWStream2  - pm.molWStream15  == 0)

    # NFE2 inlet (stream 7): no mixer, direct connection from stream 6 (NFE1 Perm)
    pm.conPass7Li = pyo.Constraint(expr=pm.molLiStream7 - pm.molLiStream6 == 0)
    pm.conPass7Co = pyo.Constraint(expr=pm.molCoStream7 - pm.molCoStream6 == 0)
    pm.conPass7W  = pyo.Constraint(expr=pm.molWStream7  - pm.molWStream6  == 0)

    # NFS2 inlet (stream 13): no mixer, direct connection from stream 11 (NFS1 Ret)
    pm.conPass13Li = pyo.Constraint(expr=pm.molLiStream13 - pm.molLiStream11 == 0)
    pm.conPass13Co = pyo.Constraint(expr=pm.molCoStream13 - pm.molCoStream11 == 0)
    pm.conPass13W  = pyo.Constraint(expr=pm.molWStream13  - pm.molWStream11  == 0)

    # Product purity / recovery targets
    pm.conLiPurity = pyo.Constraint(
        expr=pm.molLiStream9 - 0.95 * (pm.molLiStream9 + pm.molCoStream9) >= 0
    )
    pm.conCoPurity = pyo.Constraint(
        expr=pm.molCoStream14 - 0.95 * (pm.molLiStream14 + pm.molCoStream14) >= 0
    )
    pm.conLiRecovery = pyo.Constraint(
        expr=pm.molLiStream9 - 0.95 * pm.molLiFeed >= 0
    )
    pm.conCoRecovery = pyo.Constraint(
        expr=pm.molCoStream14 - 0.95 * pm.molCoFeed >= 0
    )

    # =========================================================================
    # CONCENTRATION VALIDITY CUTS (use if bound reduction is necessary)
    # =========================================================================
    # pm.conConcUBStream1Li = pyo.Constraint(
    #     expr=pm.molLiStream1 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream1 - 30000.0 * pm.molweightH2OgMol * pm.molWStream1 <= 0
    # )
    # pm.conConcUBStream1Co = pyo.Constraint(
    #     expr=pm.molCoStream1 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream1 - 30000.0 * pm.molweightH2OgMol * pm.molWStream1 <= 0
    # )
    # pm.conConcUBStream1W = pyo.Constraint(
    #     expr=pm.molWStream1 * (pm.densityH2OGM3 - 30000.0 * pm.molweightH2OgMol) - 30000.0 * pm.molweightLi * pm.molLiStream1 - 30000.0 * pm.molweightCo * pm.molCoStream1 <= 0
    # )

    # pm.conConcUBStream2Li = pyo.Constraint(
    #     expr=pm.molLiStream2 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream2 - 30000.0 * pm.molweightH2OgMol * pm.molWStream2 <= 0
    # )
    # pm.conConcUBStream2Co = pyo.Constraint(
    #     expr=pm.molCoStream2 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream2 - 30000.0 * pm.molweightH2OgMol * pm.molWStream2 <= 0
    # )

    # pm.conConcUBStream3Li = pyo.Constraint(
    #     expr=pm.molLiStream3 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream3 - 30000.0 * pm.molweightH2OgMol * pm.molWStream3 <= 0
    # )
    # pm.conConcUBStream3Co = pyo.Constraint(
    #     expr=pm.molCoStream3 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream3 - 30000.0 * pm.molweightH2OgMol * pm.molWStream3 <= 0
    # )

    # pm.conConcUBStream4Li = pyo.Constraint(
    #     expr=pm.molLiStream4 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream4 - 30000.0 * pm.molweightH2OgMol * pm.molWStream4 <= 0
    # )
    # pm.conConcUBStream4Co = pyo.Constraint(
    #     expr=pm.molCoStream4 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream4 - 30000.0 * pm.molweightH2OgMol * pm.molWStream4 <= 0
    # )

    # pm.conConcUBStream5Li = pyo.Constraint(
    #     expr=pm.molLiStream5 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream5 - 30000.0 * pm.molweightH2OgMol * pm.molWStream5 <= 0
    # )
    # pm.conConcUBStream5Co = pyo.Constraint(
    #     expr=pm.molCoStream5 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream5 - 30000.0 * pm.molweightH2OgMol * pm.molWStream5 <= 0
    # )

    # pm.conConcUBStream6Li = pyo.Constraint(
    #     expr=pm.molLiStream6 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream6 - 30000.0 * pm.molweightH2OgMol * pm.molWStream6 <= 0
    # )
    # pm.conConcUBStream6Co = pyo.Constraint(
    #     expr=pm.molCoStream6 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream6 - 30000.0 * pm.molweightH2OgMol * pm.molWStream6 <= 0
    # )

    # pm.conConcUBStream7Li = pyo.Constraint(
    #     expr=pm.molLiStream7 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream7 - 30000.0 * pm.molweightH2OgMol * pm.molWStream7 <= 0
    # )
    # pm.conConcUBStream7Co = pyo.Constraint(
    #     expr=pm.molCoStream7 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream7 - 30000.0 * pm.molweightH2OgMol * pm.molWStream7 <= 0
    # )

    # pm.conConcUBStream8Li = pyo.Constraint(
    #     expr=pm.molLiStream8 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream8 - 30000.0 * pm.molweightH2OgMol * pm.molWStream8 <= 0
    # )
    # pm.conConcUBStream8Co = pyo.Constraint(
    #     expr=pm.molCoStream8 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream8 - 30000.0 * pm.molweightH2OgMol * pm.molWStream8 <= 0
    # )

    # pm.conConcUBStream9Li = pyo.Constraint(
    #     expr=pm.molLiStream9 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream9 - 30000.0 * pm.molweightH2OgMol * pm.molWStream9 <= 0
    # )
    # pm.conConcUBStream9Co = pyo.Constraint(
    #     expr=pm.molCoStream9 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream9 - 30000.0 * pm.molweightH2OgMol * pm.molWStream9 <= 0
    # )

    # pm.conConcUBStream10Li = pyo.Constraint(
    #     expr=pm.molLiStream10 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream10 - 30000.0 * pm.molweightH2OgMol * pm.molWStream10 <= 0
    # )
    # pm.conConcUBStream10Co = pyo.Constraint(
    #     expr=pm.molCoStream10 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream10 - 30000.0 * pm.molweightH2OgMol * pm.molWStream10 <= 0
    # )

    # pm.conConcUBStream11Li = pyo.Constraint(
    #     expr=pm.molLiStream11 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream11 - 30000.0 * pm.molweightH2OgMol * pm.molWStream11 <= 0
    # )
    # pm.conConcUBStream11Co = pyo.Constraint(
    #     expr=pm.molCoStream11 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream11 - 30000.0 * pm.molweightH2OgMol * pm.molWStream11 <= 0
    # )

    # pm.conConcUBStream12Li = pyo.Constraint(
    #     expr=pm.molLiStream12 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream12 - 30000.0 * pm.molweightH2OgMol * pm.molWStream12 <= 0
    # )
    # pm.conConcUBStream12Co = pyo.Constraint(
    #     expr=pm.molCoStream12 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream12 - 30000.0 * pm.molweightH2OgMol * pm.molWStream12 <= 0
    # )

    # pm.conConcUBStream13Li = pyo.Constraint(
    #     expr=pm.molLiStream13 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream13 - 30000.0 * pm.molweightH2OgMol * pm.molWStream13 <= 0
    # )
    # pm.conConcUBStream13Co = pyo.Constraint(
    #     expr=pm.molCoStream13 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream13 - 30000.0 * pm.molweightH2OgMol * pm.molWStream13 <= 0
    # )

    # pm.conConcUBStream14Li = pyo.Constraint(
    #     expr=pm.molLiStream14 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream14 - 30000.0 * pm.molweightH2OgMol * pm.molWStream14 <= 0
    # )
    # pm.conConcUBStream14Co = pyo.Constraint(
    #     expr=pm.molCoStream14 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream14 - 30000.0 * pm.molweightH2OgMol * pm.molWStream14 <= 0
    # )

    # pm.conConcUBStream15Li = pyo.Constraint(
    #     expr=pm.molLiStream15 * (pm.densityH2OGM3 - 30000.0 * pm.molweightLi) - 30000.0 * pm.molweightCo * pm.molCoStream15 - 30000.0 * pm.molweightH2OgMol * pm.molWStream15 <= 0
    # )
    # pm.conConcUBStream15Co = pyo.Constraint(
    #     expr=pm.molCoStream15 * (pm.densityH2OGM3 - 30000.0 * pm.molweightCo) - 30000.0 * pm.molweightLi * pm.molLiStream15 - 30000.0 * pm.molweightH2OgMol * pm.molWStream15 <= 0
    # )

    # =========================================================================
    # SECOND-STAGE VARIABLES: Unit-Local NF Model Variables for Each Unit
    # =========================================================================
    pm.dP                 = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e5, 40e5), initialize=10e5)        # transmembrane pressure [Pa]
    pm.memLength          = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0.1, 200), initialize=1.0)          # membrane element length [m]
    pm.memWidth           = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0.95, 0.95), initialize=0.95)       # membrane element width [m] -- pinned to 0.95 via equal bounds (NOT .fix(): a fixed second-stage Var breaks StoModelBuilder's var2 remapping/replace_expressions, producing a ctype=None node that crashes the BARON writer; ED never fixes any var)
    pm.channelHeight      = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(2e-4, 5e-3), initialize=7e-4)       # feed spacer thickness [m]
    pm.flowOutRetND       = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-6, 100), initialize=0.5)         # retentate flow (nondim)
    pm.flowOutPermND      = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-6, 100), initialize=0.5)         # permeate flow (nondim)
    pm.cLiRetND           = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-6, 100), initialize=1.0)         # Li retentate conc (nondim)
    pm.cCoRetND           = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-6, 100), initialize=2.0)         # Co retentate conc (nondim)
    pm.cLiPermND          = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-6, 100), initialize=0.8)         # Li permeate conc (nondim)
    pm.cCoPermND          = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-6, 500), initialize=0.1)         # Co permeate conc (nondim)
    pm.Jw                 = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-8, 1e-4), initialize=5e-6)       # solvent flux [m3/(m2.s)]
    pm.fLi                = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-4, 1), initialize=0.5)           # Spiegler-Kedem F for Li
    pm.fCo                = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(1e-4, 1), initialize=0.5)           # Spiegler-Kedem F for Co

    # NOTE: memWidth is pinned to 0.95 above via equal bounds (0.95, 0.95)

    # Unit-local cost variables 
    pm.capex     = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 1e9), initialize=1e5)
    pm.opex1     = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 1e9), initialize=1e5)
    pm.opex2     = pyo.Var(pm.units, within=pyo.NonNegativeReals, bounds=(0, 1e9), initialize=1e5)

    # =========================================================================
    # EXPRESSIONS: Unit-Local Inlet Conditions from Flowsheet Interface
    # Unit inlet streams: unit 1 (NFF) <- stream 1, unit 2 (NFE1) <- stream 4,
    # unit 3 (NFE2) <- stream 7, unit 4 (NFS1) <- stream 10, unit 5 (NFS2) <- stream 13.
    # =========================================================================

    def molLiInRule(m, unitIdx):
        if unitIdx == 1:
            return m.molLiStream1
        if unitIdx == 2:
            return m.molLiStream4
        if unitIdx == 3:
            return m.molLiStream7
        if unitIdx == 4:
            return m.molLiStream10
        return m.molLiStream13
    pm.molLiIn = pyo.Expression(pm.units, rule=molLiInRule)

    def molCoInRule(m, unitIdx):
        if unitIdx == 1:
            return m.molCoStream1
        if unitIdx == 2:
            return m.molCoStream4
        if unitIdx == 3:
            return m.molCoStream7
        if unitIdx == 4:
            return m.molCoStream10
        return m.molCoStream13
    pm.molCoIn = pyo.Expression(pm.units, rule=molCoInRule)

    def molWInRule(m, unitIdx):
        if unitIdx == 1:
            return m.molWStream1
        if unitIdx == 2:
            return m.molWStream4
        if unitIdx == 3:
            return m.molWStream7
        if unitIdx == 4:
            return m.molWStream10
        return m.molWStream13
    pm.molWIn = pyo.Expression(pm.units, rule=molWInRule)

    # Mass-balance conversion mol/s -> (flow [m3/s], conc [mol/m3]) per unit,
    # same technique as mainNFMolFlow.py (dilute-solution assumption).
    def massInRule(m, unitIdx):
        return m.molLiIn[unitIdx] * m.molweightLi + m.molCoIn[unitIdx] * m.molweightCo + m.molWIn[unitIdx] * m.molweightH2OgMol
    pm.massIn = pyo.Expression(pm.units, rule=massInRule)   # g/s

    def flowInRule(m, unitIdx):
        return m.massIn[unitIdx] / m.densityH2OGM3
    pm.flowIn = pyo.Expression(pm.units, rule=flowInRule)   # m3/s

    def cLiInRule(m, unitIdx):
        return m.molLiIn[unitIdx] / m.flowIn[unitIdx]
    pm.cLiIn = pyo.Expression(pm.units, rule=cLiInRule)      # mol/m3

    def cCoInRule(m, unitIdx):
        return m.molCoIn[unitIdx] / m.flowIn[unitIdx]
    pm.cCoIn = pyo.Expression(pm.units, rule=cCoInRule)      # mol/m3

    def flowInPerElemRule(m, unitIdx):
        return m.flowIn[unitIdx] / NUM_ELEMENTS
    pm.flowInPerElem = pyo.Expression(pm.units, rule=flowInPerElemRule)

    # =========================================================================
    # UNIT MODELS: NF unit physics for each unit
    # =========================================================================

    # ---- Geometry ----
    def memAreaRule(m, unitIdx):
        return m.memLength[unitIdx] * m.memWidth[unitIdx]
    pm.memArea = pyo.Expression(pm.units, rule=memAreaRule)

    def memAreaTotalRule(m, unitIdx):
        return m.memArea[unitIdx] * NUM_ELEMENTS
    pm.memAreaTotal = pyo.Expression(pm.units, rule=memAreaTotalRule)

    # ---- Hydrodynamics ----
    def avgFlowRetRule(m, unitIdx):
        return 0.5 * (m.flowOutRetND[unitIdx] * SCALE_FAC_FLOW + m.flowInPerElem[unitIdx])
    pm.avgFlowRet = pyo.Expression(pm.units, rule=avgFlowRetRule)

    def uFeedRule(m, unitIdx):
        return m.avgFlowRet[unitIdx] / (m.memWidth[unitIdx] * m.channelHeight[unitIdx])
    pm.uFeed = pyo.Expression(pm.units, rule=uFeedRule)

    def reRule(m, unitIdx):
        return m.densityH2O * m.uFeed[unitIdx] * (2 * m.channelHeight[unitIdx]) / m.viscosity
    pm.Re = pyo.Expression(pm.units, rule=reRule)

    def scLiRule(m, unitIdx):
        return m.viscosity / (m.densityH2O * m.diffLi)
    pm.ScLi = pyo.Expression(pm.units, rule=scLiRule)

    def scCoRule(m, unitIdx):
        return m.viscosity / (m.densityH2O * m.diffCo)
    pm.ScCo = pyo.Expression(pm.units, rule=scCoRule)

    def shLiRule(m, unitIdx):
        return 1.85 * (m.Re[unitIdx] * m.ScLi[unitIdx] * 2 * m.channelHeight[unitIdx] / m.memLength[unitIdx]) ** (1.0 / 3.0)
    pm.ShLi = pyo.Expression(pm.units, rule=shLiRule)

    def shCoRule(m, unitIdx):
        return 1.85 * (m.Re[unitIdx] * m.ScCo[unitIdx] * 2 * m.channelHeight[unitIdx] / m.memLength[unitIdx]) ** (1.0 / 3.0)
    pm.ShCo = pyo.Expression(pm.units, rule=shCoRule)

    def kLiRule(m, unitIdx):
        return m.ShLi[unitIdx] * m.diffLi / (2 * m.channelHeight[unitIdx])
    pm.kLi = pyo.Expression(pm.units, rule=kLiRule)

    def kCoRule(m, unitIdx):
        return m.ShCo[unitIdx] * m.diffCo / (2 * m.channelHeight[unitIdx])
    pm.kCo = pyo.Expression(pm.units, rule=kCoRule)

    # ---- Concentration polarization ----
    def cpFactorLiRule(m, unitIdx):
        return pyo.exp(m.Jw[unitIdx] / m.kLi[unitIdx])
    pm.cpFactorLi = pyo.Expression(pm.units, rule=cpFactorLiRule)

    def cpFactorCoRule(m, unitIdx):
        return pyo.exp(m.Jw[unitIdx] / m.kCo[unitIdx])
    pm.cpFactorCo = pyo.Expression(pm.units, rule=cpFactorCoRule)

    def avgConcLiRetRule(m, unitIdx):
        return 0.5 * (m.cLiRetND[unitIdx] * SCALE_FAC_CONC + m.cLiIn[unitIdx])
    pm.avgConcLiRet = pyo.Expression(pm.units, rule=avgConcLiRetRule)

    def avgConcCoRetRule(m, unitIdx):
        return 0.5 * (m.cCoRetND[unitIdx] * SCALE_FAC_CONC + m.cCoIn[unitIdx])
    pm.avgConcCoRet = pyo.Expression(pm.units, rule=avgConcCoRetRule)

    def cLiMemRule(m, unitIdx):
        return m.avgConcLiRet[unitIdx] * m.cpFactorLi[unitIdx]
    pm.cLiMem = pyo.Expression(pm.units, rule=cLiMemRule)

    def cCoMemRule(m, unitIdx):
        return m.avgConcCoRet[unitIdx] * m.cpFactorCo[unitIdx]
    pm.cCoMem = pyo.Expression(pm.units, rule=cCoMemRule)

    # ---- Osmotic pressure ----
    def piRetEffRule(m, unitIdx):
        return m.Rg * m.temp * (m.sigmaLi * m.nuLi * m.avgConcLiRet[unitIdx] + m.sigmaCo * m.nuCo * m.avgConcCoRet[unitIdx])
    pm.piRetEff = pyo.Expression(pm.units, rule=piRetEffRule)

    def piPermEffRule(m, unitIdx):
        return m.Rg * m.temp * (m.sigmaLi * m.nuLi * m.cLiPermND[unitIdx] * SCALE_FAC_CONC + m.sigmaCo * m.nuCo * m.cCoPermND[unitIdx] * SCALE_FAC_CONC)
    pm.piPermEff = pyo.Expression(pm.units, rule=piPermEffRule)

    # ---- Limiting flux ----
    def jwCritCoRule(m, unitIdx):
        return m.kCo[unitIdx] * pyo.log(m.cpMax)
    pm.JwCritCo = pyo.Expression(pm.units, rule=jwCritCoRule)

    # ---- Pressure drop through feed channel ----
    def fFeedRule(m, unitIdx):
        return 96.0 / m.Re[unitIdx]
    pm.fFeed = pyo.Expression(pm.units, rule=fFeedRule)

    def presDropChannelRule(m, unitIdx):
        return m.densityH2O * m.fFeed[unitIdx] * m.memLength[unitIdx] * m.uFeed[unitIdx] ** 2 / (4 * m.channelHeight[unitIdx])
    pm.presDropChannel = pyo.Expression(pm.units, rule=presDropChannelRule)

    # ---- Capital cost intermediates ----
    def capexMembraneRule(m, unitIdx):
        return m.costMembrane * m.memAreaTotal[unitIdx]
    pm.capexMembrane = pyo.Expression(pm.units, rule=capexMembraneRule)

    def capexHousingRule(m, unitIdx):
        return 20.0 * m.memAreaTotal[unitIdx]
    pm.capexHousing = pyo.Expression(pm.units, rule=capexHousingRule)

    # ---- Operating cost intermediates ----
    def pumpPowerTMPRule(m, unitIdx):
        return m.dP[unitIdx] * m.flowIn[unitIdx] / m.pumpEff
    pm.pumpPowerTMP = pyo.Expression(pm.units, rule=pumpPowerTMPRule)

    def pumpPowerChannelRule(m, unitIdx):
        return m.presDropChannel[unitIdx] * m.avgFlowRet[unitIdx] * NUM_ELEMENTS / m.pumpEff
    pm.pumpPowerChannel = pyo.Expression(pm.units, rule=pumpPowerChannelRule)

    # ---- Product value (reporting) ----
    def coRecoveredTonYrRule(m, unitIdx):
        return (m.cCoRetND[unitIdx] * SCALE_FAC_CONC * m.flowOutRetND[unitIdx] * SCALE_FAC_FLOW
                * NUM_ELEMENTS * 58.933e-3 * 86400 * m.daysOperation * 0.001)
    pm.coRecoveredTonYr = pyo.Expression(pm.units, rule=coRecoveredTonYrRule)

    def liRecoveredTonYrRule(m, unitIdx):
        return (m.cLiPermND[unitIdx] * SCALE_FAC_CONC * m.flowOutPermND[unitIdx] * SCALE_FAC_FLOW
                * NUM_ELEMENTS * 6.941e-3 * 86400 * m.daysOperation * 0.001)
    pm.liRecoveredTonYr = pyo.Expression(pm.units, rule=liRecoveredTonYrRule)

    ######################################################################################################
    # CONSTRUCT CONSTRAINTS 
    ######################################################################################################

    def constraint1Rule(m, unitIdx):
        return (m.flowOutPermND[unitIdx] + m.flowOutRetND[unitIdx]
                - m.flowInPerElem[unitIdx] / SCALE_FAC_FLOW) == 0
    pm.constraint1 = pyo.Constraint(pm.units, rule=constraint1Rule)

    def constraint2Rule(m, unitIdx):
        return (m.flowOutPermND[unitIdx] * m.cLiPermND[unitIdx]
                + m.flowOutRetND[unitIdx] * m.cLiRetND[unitIdx]
                - (m.flowInPerElem[unitIdx] * m.cLiIn[unitIdx]) / (SCALE_FAC_FLOW * SCALE_FAC_CONC)) == 0
    pm.constraint2 = pyo.Constraint(pm.units, rule=constraint2Rule)

    def constraint3Rule(m, unitIdx):
        return (m.flowOutPermND[unitIdx] * m.cCoPermND[unitIdx]
                + m.flowOutRetND[unitIdx] * m.cCoRetND[unitIdx]
                - (m.flowInPerElem[unitIdx] * m.cCoIn[unitIdx]) / (SCALE_FAC_FLOW * SCALE_FAC_CONC)) == 0
    pm.constraint3 = pyo.Constraint(pm.units, rule=constraint3Rule)

    def constraint4Rule(m, unitIdx):
        return m.Jw[unitIdx] - m.Lp * (m.dP[unitIdx] - (m.piRetEff[unitIdx] - m.piPermEff[unitIdx])) == 0
    pm.constraint4 = pyo.Constraint(pm.units, rule=constraint4Rule)

    def constraint5Rule(m, unitIdx):
        return m.flowOutPermND[unitIdx] * SCALE_FAC_FLOW - m.Jw[unitIdx] * m.memArea[unitIdx] == 0
    pm.constraint5 = pyo.Constraint(pm.units, rule=constraint5Rule)

    def constraint6Rule(m, unitIdx):
        return m.fLi[unitIdx] - pyo.exp(-m.Jw[unitIdx] * (1 - m.sigmaLi) / m.pLi) == 0
    pm.constraint6 = pyo.Constraint(pm.units, rule=constraint6Rule)

    def constraint7Rule(m, unitIdx):
        return m.fCo[unitIdx] - pyo.exp(-m.Jw[unitIdx] * (1 - m.sigmaCo) / m.pCo) == 0
    pm.constraint7 = pyo.Constraint(pm.units, rule=constraint7Rule)

    def constraint8Rule(m, unitIdx):
        return (m.cLiPermND[unitIdx] * SCALE_FAC_CONC * (1 - m.sigmaLi * m.fLi[unitIdx])
                - (1 - m.sigmaLi) * m.cLiMem[unitIdx]) == 0
    pm.constraint8 = pyo.Constraint(pm.units, rule=constraint8Rule)

    def constraint9Rule(m, unitIdx):
        return (m.cCoPermND[unitIdx] * SCALE_FAC_CONC * (1 - m.sigmaCo * m.fCo[unitIdx])
                - (1 - m.sigmaCo) * m.cCoMem[unitIdx]) == 0
    pm.constraint9 = pyo.Constraint(pm.units, rule=constraint9Rule)

    def constraint10Rule(m, unitIdx):
        return m.Jw[unitIdx] / m.JwCritCo[unitIdx] - 1 <= 0
    pm.constraint10 = pyo.Constraint(pm.units, rule=constraint10Rule)

    def conCapexRule(m, unitIdx):
        return m.capex[unitIdx] == (m.capexMembrane[unitIdx] + m.capexHousing[unitIdx])/5000
    pm.conCapex = pyo.Constraint(pm.units, rule=conCapexRule)

    def conOpex1Rule(m, unitIdx):
        return m.opex1[unitIdx] == (m.costElec * m.pumpPowerTMP[unitIdx] * (24 * m.daysOperation) / 1000)/5000
    pm.conOpex1 = pyo.Constraint(pm.units, rule=conOpex1Rule)

    def conOpex2Rule(m, unitIdx):
        return m.opex2[unitIdx] == (m.costElec * m.pumpPowerChannel[unitIdx] * (24 * m.daysOperation) / 1000)/5000
    pm.conOpex2 = pyo.Constraint(pm.units, rule=conOpex2Rule)

    # =========================================================================
    # Convert unit outlet ND vars to molar flows for flowsheet-level linking
    # =========================================================================
    # --- unit 1 outlets: stream 2 (Ret), stream 3 (Perm) ---
    pm.flowStream2 = pyo.Expression(expr=pm.flowOutRetND[1] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream2  = pyo.Expression(expr=pm.cLiRetND[1] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream2  = pyo.Expression(expr=pm.cCoRetND[1] * SCALE_FAC_CONC)   # mol/m3
    pm.flowStream3 = pyo.Expression(expr=pm.flowOutPermND[1] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream3  = pyo.Expression(expr=pm.cLiPermND[1] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream3  = pyo.Expression(expr=pm.cCoPermND[1] * SCALE_FAC_CONC)   # mol/m3

    # --- unit 2 outlets: stream 5 (Ret), stream 6 (Perm) ---
    pm.flowStream5 = pyo.Expression(expr=pm.flowOutRetND[2] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream5  = pyo.Expression(expr=pm.cLiRetND[2] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream5  = pyo.Expression(expr=pm.cCoRetND[2] * SCALE_FAC_CONC)   # mol/m3
    pm.flowStream6 = pyo.Expression(expr=pm.flowOutPermND[2] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream6  = pyo.Expression(expr=pm.cLiPermND[2] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream6  = pyo.Expression(expr=pm.cCoPermND[2] * SCALE_FAC_CONC)   # mol/m3

    # --- unit 3 outlets: stream 8 (Ret), stream 9 (Perm) ---
    pm.flowStream8 = pyo.Expression(expr=pm.flowOutRetND[3] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream8  = pyo.Expression(expr=pm.cLiRetND[3] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream8  = pyo.Expression(expr=pm.cCoRetND[3] * SCALE_FAC_CONC)   # mol/m3
    pm.flowStream9 = pyo.Expression(expr=pm.flowOutPermND[3] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream9  = pyo.Expression(expr=pm.cLiPermND[3] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream9  = pyo.Expression(expr=pm.cCoPermND[3] * SCALE_FAC_CONC)   # mol/m3

    # --- unit 4 outlets: stream 11 (Ret), stream 12 (Perm) ---
    pm.flowStream11 = pyo.Expression(expr=pm.flowOutRetND[4] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream11  = pyo.Expression(expr=pm.cLiRetND[4] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream11  = pyo.Expression(expr=pm.cCoRetND[4] * SCALE_FAC_CONC)   # mol/m3
    pm.flowStream12 = pyo.Expression(expr=pm.flowOutPermND[4] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream12  = pyo.Expression(expr=pm.cLiPermND[4] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream12  = pyo.Expression(expr=pm.cCoPermND[4] * SCALE_FAC_CONC)   # mol/m3

    # --- unit 5 outlets: stream 14 (Ret), stream 15 (Perm) ---
    pm.flowStream14 = pyo.Expression(expr=pm.flowOutRetND[5] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream14  = pyo.Expression(expr=pm.cLiRetND[5] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream14  = pyo.Expression(expr=pm.cCoRetND[5] * SCALE_FAC_CONC)   # mol/m3
    pm.flowStream15 = pyo.Expression(expr=pm.flowOutPermND[5] * SCALE_FAC_FLOW * NUM_ELEMENTS)   # m3/s
    pm.cLiStream15  = pyo.Expression(expr=pm.cLiPermND[5] * SCALE_FAC_CONC)   # mol/m3
    pm.cCoStream15  = pyo.Expression(expr=pm.cCoPermND[5] * SCALE_FAC_CONC)   # mol/m3

    # =========================================================================
    # LINKING CONSTRAINTS: Connect Broken Streams to Unit Outlets
    # =========================================================================
    def linkStream2LiRule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molLiStream2 == m.cLiStream2 * m.flowStream2
    pm.conLinkStream2Li = pyo.Constraint(pm.units, rule=linkStream2LiRule)

    def linkStream2CoRule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molCoStream2 == m.cCoStream2 * m.flowStream2
    pm.conLinkStream2Co = pyo.Constraint(pm.units, rule=linkStream2CoRule)

    def linkStream2WRule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molWStream2 == m.flowStream2 * (m.densityH2OGM3 - m.cLiStream2 * m.molweightLi - m.cCoStream2 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream2W = pyo.Constraint(pm.units, rule=linkStream2WRule)

    def linkStream3LiRule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molLiStream3 == m.cLiStream3 * m.flowStream3
    pm.conLinkStream3Li = pyo.Constraint(pm.units, rule=linkStream3LiRule)

    def linkStream3CoRule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molCoStream3 == m.cCoStream3 * m.flowStream3
    pm.conLinkStream3Co = pyo.Constraint(pm.units, rule=linkStream3CoRule)

    def linkStream3WRule(m, unitIdx):
        if unitIdx != 1:
            return pyo.Constraint.Skip
        return m.molWStream3 == m.flowStream3 * (m.densityH2OGM3 - m.cLiStream3 * m.molweightLi - m.cCoStream3 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream3W = pyo.Constraint(pm.units, rule=linkStream3WRule)

    def linkStream5LiRule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molLiStream5 == m.cLiStream5 * m.flowStream5
    pm.conLinkStream5Li = pyo.Constraint(pm.units, rule=linkStream5LiRule)

    def linkStream5CoRule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molCoStream5 == m.cCoStream5 * m.flowStream5
    pm.conLinkStream5Co = pyo.Constraint(pm.units, rule=linkStream5CoRule)

    def linkStream5WRule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molWStream5 == m.flowStream5 * (m.densityH2OGM3 - m.cLiStream5 * m.molweightLi - m.cCoStream5 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream5W = pyo.Constraint(pm.units, rule=linkStream5WRule)

    def linkStream6LiRule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molLiStream6 == m.cLiStream6 * m.flowStream6
    pm.conLinkStream6Li = pyo.Constraint(pm.units, rule=linkStream6LiRule)

    def linkStream6CoRule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molCoStream6 == m.cCoStream6 * m.flowStream6
    pm.conLinkStream6Co = pyo.Constraint(pm.units, rule=linkStream6CoRule)

    def linkStream6WRule(m, unitIdx):
        if unitIdx != 2:
            return pyo.Constraint.Skip
        return m.molWStream6 == m.flowStream6 * (m.densityH2OGM3 - m.cLiStream6 * m.molweightLi - m.cCoStream6 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream6W = pyo.Constraint(pm.units, rule=linkStream6WRule)

    def linkStream8LiRule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molLiStream8 == m.cLiStream8 * m.flowStream8
    pm.conLinkStream8Li = pyo.Constraint(pm.units, rule=linkStream8LiRule)

    def linkStream8CoRule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molCoStream8 == m.cCoStream8 * m.flowStream8
    pm.conLinkStream8Co = pyo.Constraint(pm.units, rule=linkStream8CoRule)

    def linkStream8WRule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molWStream8 == m.flowStream8 * (m.densityH2OGM3 - m.cLiStream8 * m.molweightLi - m.cCoStream8 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream8W = pyo.Constraint(pm.units, rule=linkStream8WRule)

    def linkStream9LiRule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molLiStream9 == m.cLiStream9 * m.flowStream9
    pm.conLinkStream9Li = pyo.Constraint(pm.units, rule=linkStream9LiRule)

    def linkStream9CoRule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molCoStream9 == m.cCoStream9 * m.flowStream9
    pm.conLinkStream9Co = pyo.Constraint(pm.units, rule=linkStream9CoRule)

    def linkStream9WRule(m, unitIdx):
        if unitIdx != 3:
            return pyo.Constraint.Skip
        return m.molWStream9 == m.flowStream9 * (m.densityH2OGM3 - m.cLiStream9 * m.molweightLi - m.cCoStream9 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream9W = pyo.Constraint(pm.units, rule=linkStream9WRule)

    def linkStream11LiRule(m, unitIdx):
        if unitIdx != 4:
            return pyo.Constraint.Skip
        return m.molLiStream11 == m.cLiStream11 * m.flowStream11
    pm.conLinkStream11Li = pyo.Constraint(pm.units, rule=linkStream11LiRule)

    def linkStream11CoRule(m, unitIdx):
        if unitIdx != 4:
            return pyo.Constraint.Skip
        return m.molCoStream11 == m.cCoStream11 * m.flowStream11
    pm.conLinkStream11Co = pyo.Constraint(pm.units, rule=linkStream11CoRule)

    def linkStream11WRule(m, unitIdx):
        if unitIdx != 4:
            return pyo.Constraint.Skip
        return m.molWStream11 == m.flowStream11 * (m.densityH2OGM3 - m.cLiStream11 * m.molweightLi - m.cCoStream11 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream11W = pyo.Constraint(pm.units, rule=linkStream11WRule)

    def linkStream12LiRule(m, unitIdx):
        if unitIdx != 4:
            return pyo.Constraint.Skip
        return m.molLiStream12 == m.cLiStream12 * m.flowStream12
    pm.conLinkStream12Li = pyo.Constraint(pm.units, rule=linkStream12LiRule)

    def linkStream12CoRule(m, unitIdx):
        if unitIdx != 4:
            return pyo.Constraint.Skip
        return m.molCoStream12 == m.cCoStream12 * m.flowStream12
    pm.conLinkStream12Co = pyo.Constraint(pm.units, rule=linkStream12CoRule)

    def linkStream12WRule(m, unitIdx):
        if unitIdx != 4:
            return pyo.Constraint.Skip
        return m.molWStream12 == m.flowStream12 * (m.densityH2OGM3 - m.cLiStream12 * m.molweightLi - m.cCoStream12 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream12W = pyo.Constraint(pm.units, rule=linkStream12WRule)

    def linkStream14LiRule(m, unitIdx):
        if unitIdx != 5:
            return pyo.Constraint.Skip
        return m.molLiStream14 == m.cLiStream14 * m.flowStream14
    pm.conLinkStream14Li = pyo.Constraint(pm.units, rule=linkStream14LiRule)

    def linkStream14CoRule(m, unitIdx):
        if unitIdx != 5:
            return pyo.Constraint.Skip
        return m.molCoStream14 == m.cCoStream14 * m.flowStream14
    pm.conLinkStream14Co = pyo.Constraint(pm.units, rule=linkStream14CoRule)

    def linkStream14WRule(m, unitIdx):
        if unitIdx != 5:
            return pyo.Constraint.Skip
        return m.molWStream14 == m.flowStream14 * (m.densityH2OGM3 - m.cLiStream14 * m.molweightLi - m.cCoStream14 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream14W = pyo.Constraint(pm.units, rule=linkStream14WRule)

    def linkStream15LiRule(m, unitIdx):
        if unitIdx != 5:
            return pyo.Constraint.Skip
        return m.molLiStream15 == m.cLiStream15 * m.flowStream15
    pm.conLinkStream15Li = pyo.Constraint(pm.units, rule=linkStream15LiRule)

    def linkStream15CoRule(m, unitIdx):
        if unitIdx != 5:
            return pyo.Constraint.Skip
        return m.molCoStream15 == m.cCoStream15 * m.flowStream15
    pm.conLinkStream15Co = pyo.Constraint(pm.units, rule=linkStream15CoRule)

    def linkStream15WRule(m, unitIdx):
        if unitIdx != 5:
            return pyo.Constraint.Skip
        return m.molWStream15 == m.flowStream15 * (m.densityH2OGM3 - m.cLiStream15 * m.molweightLi - m.cCoStream15 * m.molweightCo) / m.molweightH2OgMol
    pm.conLinkStream15W = pyo.Constraint(pm.units, rule=linkStream15WRule)

    if returnPyomoModel:
        return pm

    # =========================================================================
    # OBJECTIVE: Minimize total cost across all units
    # =========================================================================
    builder = StoModelBuilder('pyomo', name='NFUnit', m_type='NLP', hint=False)
    scenarios = list(pm.units)

    var1_names = [
        "molLiStream1", "molCoStream1", "molWStream1",
        "molLiStream2", "molCoStream2", "molWStream2",
        "molLiStream3", "molCoStream3", "molWStream3",
        "molLiStream4", "molCoStream4", "molWStream4",
        "molLiStream5", "molCoStream5", "molWStream5",
        "molLiStream6", "molCoStream6", "molWStream6",
        "molLiStream7", "molCoStream7", "molWStream7",
        "molLiStream8", "molCoStream8", "molWStream8",
        "molLiStream9", "molCoStream9", "molWStream9",
        "molLiStream10", "molCoStream10", "molWStream10",
        "molLiStream11", "molCoStream11", "molWStream11",
        "molLiStream12", "molCoStream12", "molWStream12",
        "molLiStream13", "molCoStream13", "molWStream13",
        "molLiStream14", "molCoStream14", "molWStream14",
        "molLiStream15", "molCoStream15", "molWStream15",
    ]
    con1_names = [
        "conMixer1Li", "conMixer1Co", "conMixer1W",
        "conMixer4Li", "conMixer4Co", "conMixer4W",
        "conMixer10Li", "conMixer10Co", "conMixer10W",
        "conPass7Li", "conPass7Co", "conPass7W",
        "conPass13Li", "conPass13Co", "conPass13W",
        "conLiPurity", "conCoPurity", "conLiRecovery", "conCoRecovery",
        # "conConcUBStream1Li", "conConcUBStream1Co", "conConcUBStream1W",
        # "conConcUBStream2Li", "conConcUBStream2Co", "conConcUBStream2W",
        # "conConcUBStream3Li", "conConcUBStream3Co", "conConcUBStream3W",
        # "conConcUBStream4Li", "conConcUBStream4Co", "conConcUBStream4W",
        # "conConcUBStream5Li", "conConcUBStream5Co", "conConcUBStream5W",
        # "conConcUBStream6Li", "conConcUBStream6Co", "conConcUBStream6W",
        # "conConcUBStream7Li", "conConcUBStream7Co", "conConcUBStream7W",
        # "conConcUBStream8Li", "conConcUBStream8Co", "conConcUBStream8W",
        # "conConcUBStream9Li", "conConcUBStream9Co", "conConcUBStream9W",
        # "conConcUBStream10Li", "conConcUBStream10Co", "conConcUBStream10W",
        # "conConcUBStream11Li", "conConcUBStream11Co", "conConcUBStream11W",
        # "conConcUBStream12Li", "conConcUBStream12Co", "conConcUBStream12W",
        # "conConcUBStream13Li", "conConcUBStream13Co", "conConcUBStream13W",
        # "conConcUBStream14Li", "conConcUBStream14Co", "conConcUBStream14W",
        # "conConcUBStream15Li", "conConcUBStream15Co", "conConcUBStream15W",
    ]

    def _obj(m, unitIdx):
        return (m.capex[unitIdx] + m.opex1[unitIdx] + m.opex2[unitIdx])

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