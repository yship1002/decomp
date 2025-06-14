from src.analyses.convergence_analysis import HausdorffAnalyzer
from src.utility.plot import plot_converge_order
from NSPLIB.src.instances.illustrative_examples.fig12 import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
import numpy as np
from pyomo.opt import SolverFactory
import copy
sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()

from src.analyses.convergence_analysis import HausdorffAnalyzer
from src.utility.plot import plot_converge_order

alg = LagrangeanAlgo(m,lag_iter=5, solver='baron')
ca = HausdorffAnalyzer(alg)

eps_min = -3
eps_max = -3
steps = 1

tol = 1e-30
y_val={'y': 2.6199679213876177}
eps_list0, distances0= ca.analyze(y=y_val.copy(), y_optimal=True,v=-0.14585646148681208, eps_min=eps_min, eps_max=eps_max, steps=steps, tol=tol)