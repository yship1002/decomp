from NSPLIB.src.instances.EDUnit.EDUnits import const_model

from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
import cProfile
import numpy as np
from pyomo.environ import *

sto_m = const_model()
m = CaoZavalaModel.from_sto_m(sto_m)
m.build()

from pyomo.opt import SolverFactory
solver=SolverFactory("baron",executable="/Users/jyang872/Desktop/baron-osxarm64/baron")

solver=SolverFactory("ipopt")

solver.solve(m.origin_model,tee=True)
#alg=CaoZavalaAlgo(m,solver="baron")
#alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-2,ubd_midpt_fix=1,ubd_local_solve=1)