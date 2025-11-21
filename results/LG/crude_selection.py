from NSPLIB.src.instances.crude_selection.crude_selection import const_model
from src.models.lagrangean_model import LagrangeanAlgo,LagrangeanModel
import dill
crude_obj = -18350.146929611754
crude_sol = {f'crudeQuantity[{i}]': 0. for i in range(1, 10 + 1)}
crude_sol['crudeQuantity[2]'] = 150.87595641747944
crude_sol['crudeQuantity[3]'] = 201.29570746971186
crude_sol['crudeQuantity[4]'] = 56.18456149457359
crude_sol['crudeQuantity[8]'] = 162.2466500589715
crude_sol['crudeQuantity[10]'] = 18.848116800048512
crude_y_bound = {
    'crudeQuantity[1]': [0, 0],
    'crudeQuantity[2]': [150, 151],
    'crudeQuantity[3]': [201, 202],
    'crudeQuantity[4]': [56, 57],
    'crudeQuantity[5]': [0, 0],
    'crudeQuantity[6]': [0, 0],
    'crudeQuantity[7]': [0, 0],
    'crudeQuantity[8]': [162, 163],
    'crudeQuantity[9]': [0, 0],
    'crudeQuantity[10]': [18, 19]
}
sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()
binary_ys = [f'pickCrude[{i}]' for i in range(1, 10 + 1)]
binary_y_val = {y: 0 for y in binary_ys}
binary_y_val['pickCrude[2]'] = 1
binary_y_val['pickCrude[3]'] = 1
binary_y_val['pickCrude[4]'] = 1
binary_y_val['pickCrude[8]'] = 1
binary_y_val['pickCrude[10]'] = 1
m.fix_binary_y(binary_y_val)
m.update_y_bound(crude_y_bound)
alg = LagrangeanAlgo(m,lag_iter=0, solver='gurobi')
alg.solve(max_iter=1e5, max_time=3600*24,tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=1,ubd_provided=crude_obj,inherit_multiplier=True,aug_lag=True,aug_lag_iter=2)
with open('/Users/jyang872/Desktop/decomp/results/LG/crude_selection_aug.pkl', 'wb') as f:
    dill.dump(alg, f)