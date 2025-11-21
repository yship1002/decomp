from src.models.lagrangean_model import LagrangeanAlgo,LagrangeanModel
import dill
from NSPLIB.src.instances.ex8_4_4.ex8_4_4_SP import const_model

ex844_obj= 0.33272388311799445
ex844_sol = {
    'x10': -0.23129676903369037, 'x11': 1.2249985914391361, 'x12': 0.05263738880099749, 'x13': 0.5614202118628403,
    'x14': 0.6217131656002401, 'x15': 1.1, 'x16': 0.806868640168786, 'x17': 0.663161467338442, 'x6': -1.1370129865018144,
    'x7': 1.5979758364259327, 'x8': -0.5986905053355759, 'x9': 1.372365581205024
}
ex844_y_bound = {
    'x10': [-1.2, 0.8], 'x11': [0.1, 2.1], 'x12': [-1.1, 0.9], 'x13': [0, 1],
    'x14': [0, 1], 'x15': [1.1, 1.3], 'x16': [0, 1], 'x17': [0, 1],
    'x6': [-2, 0], 'x7': [0.5, 2.5], 'x8': [-1.5, 0.5], 'x9': [0.2, 2.2]
}
ex844_y_bound = {
    'x10': [-0.3, -0.2], 'x11': [1.2, 1.3], 'x12': [0, 0.1], 'x13': [0.5, 0.6],
    'x14': [0.6, 0.7], 'x15': [1.1, 1.3], 'x16': [0.8, 0.9], 'x17': [0.6, 0.7],
    'x6': [-1.2, -1.1], 'x7': [1.5, 1.6], 'x8': [-0.6, -0.5], 'x9': [1.3, 1.4]
}
sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()
#m.update_y_bound(ex844_y_bound)
alg = LagrangeanAlgo(m,lag_iter=0, solver='gurobi')
alg.solve(max_iter=1e5, max_time=3600*24,tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=1,ubd_provided=ex844_obj,inherit_multiplier=True,aug_lag=True,aug_lag_iter=2)
with open('/Users/jyang872/Desktop/decomp/results/LG/ex844_aug.pkl', 'wb') as f:
    dill.dump(alg, f)
