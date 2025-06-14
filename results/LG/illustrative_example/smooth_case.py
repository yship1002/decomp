from NSPLIB.src.instances.illustrative_examples.nonlinear_3 import const_model
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo

sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()
from src.analyses.convergence_analysis import HausdorffAnalyzer
from src.utility.plot import plot_converge_order

alg = LagrangeanAlgo(m,lag_iter=0, solver='baron')
ca = HausdorffAnalyzer(alg)

eps_min = -3
eps_max = -3
steps = 1

tol = 1e-30
y_val={0: 6.531128925558817}
eps_list, distances= ca.analyze(y=y_val.copy(), v=-113.8094952092636,y_optimal=True, eps_min=eps_min, eps_max=eps_max, steps=steps, tol=tol)