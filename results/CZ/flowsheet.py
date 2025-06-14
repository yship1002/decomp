from NSPLIB.src.instances.flowsheet.flowsheet import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
sto_m = const_model()
m = CaoZavalaModel.from_sto_m(sto_m)
m.build()
alg = CaoZavalaAlgo(m, solver='baron')
alg.solve()