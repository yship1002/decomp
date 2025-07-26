from src.analyses.convergence_analysis import HausdorffAnalyzer
from NSPLIB.src.instances.illustrative_examples.fig12 import const_model
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
import copy
sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()

alg = LagrangeanAlgo(m,lag_iter=10, solver='baron')
ca = HausdorffAnalyzer(alg)

eps_min = -3
eps_max = -1
steps = 3

tol = 1e-3
y_val={0: 2.6199679213876177}
eps_list, distances= ca.analyze(y=y_val.copy(), y_optimal=True,v=-0.14586, eps_min=eps_min, eps_max=eps_max, steps=steps, tol=tol)
eps_list, distances