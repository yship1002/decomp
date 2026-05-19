from NSPLIB.src.instances.EDUnit.EDUnits import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
import cProfile
import numpy as np
from pyomo.environ import *
# UBD:106451 0.303787+0 m = const_model()
sto_m = const_model()
m = CaoZavalaModel.from_sto_m(sto_m)
m.build()
new_bound={
  "molNStream1": [0.0,2.75],
    "molWStream1": [375.0000, 500.0000],
  "molNStream2": [0.0000, 3.0000],
  "molWStream2": [375.0000, 500.0000],
  "molNStream3": [0.0000, 3.2500],
  "molWStream3": [0.0000, 250.0000],
    "molNStream4": [0.0000, 7.0000],
    "molWStream4": [250.0000, 500.0000],
    "molNStream5": [7.5000, 15.0000],
    "molWStream5": [0.0000, 500.0000],
    "molNStream6": [0.0000, 16.0000],
    "molWStream6": [0.0000, 500.0000],
    "molNStream7": [0.0000, 17.0000],
    "molWStream7": [0.0000, 500.0000],
    "molNStream8": [0.0000, 18.0000],
    "molWStream8": [0.0000, 500.0000]
}
#m.update_y_bound(new_bound)
alg=CaoZavalaAlgo(m,solver="baron")
alg.solve(max_iter=1e5, max_time=3600*24,ubd_provided=57608)
   
# tol=1e-3
# from pyomo.opt import SolverFactory
# solver=SolverFactory("baron",options = {'EpsA': 1e-6, 'TDo':0,"AbsConsFeasTol":1e-3})
# solver.solve(m.origin_model,tee=True,keepfiles=True)
