from NSPLIB.src.instances.CHP_sizing.CHP_sizing_SP import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
sto_m = const_model()
m = CaoZavalaModel.from_sto_m(sto_m)
m.build()
# from pyomo.opt import SolverFactory
# solver = SolverFactory("baron")
# results=solver.solve(m.origin_model, tee=True)
# results.solver.time

alg=CaoZavalaAlgo(m,solver="baron")
alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-2,ubd_midpt_fix=1,ubd_local_solve=1)