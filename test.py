from NSPLIB.src.instances.process.process_SP import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()
alg = LagrangeanAlgo(m, solver='baron')

alg.solve(lag_iter=50, tol=1e-3)