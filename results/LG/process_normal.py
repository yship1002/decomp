from NSPLIB.src.instances.process.process_SP import const_model
from src.models.lagrangean_model import LagrangeanAlgo,LagrangeanModel
import dill
process_obj = -1126.4218270121305
process_sol = {"x1":1727.2601809997955,"x2":16000,"x3":104.23841082714829,"x5":2000}
process_y_bound = {"x1":[10,2000],"x2":[0,16000],"x3":[0,120],"x5":[0,2000]}
process_y_bound = {"x1":[1725,1730],"x2":[15999,16000],"x3":[100,105],"x5":[1999,2000]}
sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()
m.update_y_bound(process_y_bound)
alg = LagrangeanAlgo(m,lag_iter=3, solver='gurobi')
options = {
    'max_iter': 1e5,
    'max_time': 3600 * 24,
    'tol': 1e-3,
    'ubd_midpt_fix': 1,
    'ubd_local_solve': 1,
    'ubd_provided': process_obj,
    'inherit_multiplier': True,
    'aug_lag': False,
    'aug_lag_iter': 3,
    "aug_lag_p":1
}
alg.solve(**options)
with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/decomp/results/LG/process_normal.pkl', 'wb') as f:
    dill.dump(alg, f)