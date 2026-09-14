from NSPLIB.src.instances.EDUnit.EDUnits import const_model

from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
import cProfile
import numpy as np
from pyomo.environ import *
# UBD:57608.7

sto_m = const_model()
m = CaoZavalaModel.from_sto_m(sto_m)
m.build()
eps=10
#        Key         : Lower : Value                : Upper : Fixed : Stale : Domain
        # molNStream1 : 1e-05 :  0.22285463857648757 :    11 : False : False : NonNegativeReals
        # molNStream2 : 1e-05 : 0.061479954752751274 :    12 : False : False : NonNegativeReals
        # molNStream3 : 1e-05 :   0.1613746838237363 :    13 : False : False : NonNegativeReals
        # molNStream4 : 1e-05 :                 14.0 :    14 : False : False : NonNegativeReals
        # molNStream5 : 1e-05 :   13.994493502551212 :    15 : False : False : NonNegativeReals
        # molNStream6 : 1e-05 :  0.00550649744878918 :    16 : False : False : NonNegativeReals
        # molNStream7 : 1e-05 :  0.15586818637494734 :    17 : False : False : NonNegativeReals
        # molNStream8 : 1e-05 :   13.838625316176264 :    18 : False : False : NonNegativeReals
        # molWStream1 : 1e-05 :   123.68491498701013 :   500 : False : False : NonNegativeReals
        # molWStream2 : 1e-05 :   119.50012464909804 :   500 : False : False : NonNegativeReals
        # molWStream3 : 1e-05 :    4.184790337912115 :   500 : False : False : NonNegativeReals
        # molWStream4 : 1e-05 :   363.13431969472566 :   500 : False : False : NonNegativeReals
        # molWStream5 : 1e-05 :   362.98906121735627 :   500 : False : False : NonNegativeReals
        # molWStream6 : 1e-05 :  0.14525847736943553 :   500 : False : False : NonNegativeReals
        # molWStream7 : 1e-05 :    4.039531860542781 :   500 : False : False : NonNegativeReals
        # molWStream8 : 1e-05 :   358.94952935681357 :   500 : False : False : NonNegativeReals

# m.update_y_bound(EDUnits_sol)

tol=1e-3
# from pyomo.opt import SolverFactory
# for i in range(3):
#     solver=SolverFactory("baron",options = {'EpsA': 1e-3, 'TDo':0,"AbsConFeasTol":1e-2})
#     solver.solve(m.aux_models["lbd"][i+1],tee=True) 
# solver=SolverFactory("baron",options = {'EpsA': 1e-3, 'TDo':0,"AbsConFeasTol":1e-2})
# solver.solve(m.origin_model,tee=True)


alg=CaoZavalaAlgo(m,solver="baron")
alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_provided=57608.7)
