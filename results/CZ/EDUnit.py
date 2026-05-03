from NSPLIB.src.instances.EDUnit.EDUnits import const_model

from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
import cProfile
import numpy as np
from pyomo.environ import *
# UBD:106451

sto_m = const_model()
m = CaoZavalaModel.from_sto_m(sto_m)
m.build()
# eps=0
# EDUnits_obj=106.451
# EDUnits_sol={
#    "concStream3": [100-eps,100+eps],
#    "concStream5": [97.52475600962396-eps,97.52475600962396+eps],
#     "concStream6": [93.33645327361674-eps,93.33645327361674+eps],
#     "concStream8": [64.39428575130025-eps,64.39428575130025+eps],
#     "flowStream3": [0.022359190491272524-eps,0.022359190491272524+eps],
#     "flowStream5": [0.004560651742154918 -eps, 0.004560651742154918 +eps],
#     "flowStream6": [0.022286458333333335-eps,0.022286458333333335+eps],
#     "flowStream8": [0.004487919584215729 -eps,0.004487919584215729 +eps]
# }
# m.update_y_bound(EDUnits_sol)

# tol=1e-3
# from pyomo.opt import SolverFactory
# solver=SolverFactory("baron",options = {'EpsA': 100*tol, 'AbsConFeasTol': 1*tol, 'TDo':0, 'MDo':0,'OBTTDo':1})
# solver.solve(m.origin_model,tee=True)


alg=CaoZavalaAlgo(m,solver="baron")
alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_provided=106.451)
