from src.models.lagrangean_model import LagrangeanAlgo,LagrangeanModel
from NSPLIB.src.instances.tanksize.tanksize import const_model
import dill
tanksize_obj = 0.9030394623070541
tanksize_sol = {"productTankSize[1]": 659.2053849850757, "productTankSize[2]": 552.3954188143167,
                 "productTankSize[3]": 225.73997707963284}
tanksize_y_bound = {"productTankSize[1]": [643, 4018.36], "productTankSize[2]": [536, 3348], 
                    "productTankSize[3]": [214, 1339.45]}
tanksize_y_bound = {"productTankSize[1]": [659, 660], "productTankSize[2]": [552, 553], 
                    "productTankSize[3]": [225, 226]}
sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()
m.update_y_bound(tanksize_y_bound)
alg=LagrangeanAlgo(m,solver="baron",lag_iter=0)
alg.solve(max_iter=1e5, max_time=3600*24,tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=1,ubd_provided=tanksize_obj,inherit_multiplier=True,aug_lag=True,aug_lag_iter=2)
with open('/Users/jyang872/Desktop/decomp/results/LG/tanksize_aug.pkl', 'wb') as f:
    dill.dump(alg, f)