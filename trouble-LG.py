from graphing.helper import convergence_analysis
from src.models.cz_model import CaoZavalaAlgo, CaoZavalaModel
from NSPLIB.src.instances.process.process_SP import const_model
import numpy as np
from src.models.bb_node import BranchBoundNode
process_obj = -1126.4218270121305
process_sol = {"x1":1727.2601809997955,"x2":16000,"x3":104.23841082714829,"x5":2000}
process_y_bound = {"x1":[10,2000],"x2":[0,16000],"x3":[0,120],"x5":[0,2000]}

sto_m = const_model()
m = CaoZavalaModel.from_sto_m(sto_m)
m.build()
m.update_y_bound(process_y_bound)
alg=CaoZavalaAlgo(m,solver="gurobi")
alg.solve(max_iter=1e5, max_time=3600*24, tol=1e-4,ubd_local_solve=1,ubd_midpt_fix=0,ubd_provided=process_obj)
