from src.models.lagrangean_model import LagrangeanAlgo,LagrangeanModel
from NSPLIB.src.instances.crude_selection.crude_selection import const_model
import dill
crude_obj = -18350.146929611754
crude_sol = {f'crudeQuantity[{i}]': 0. for i in range(1, 10 + 1)}
crude_sol['crudeQuantity[2]'] = 150.87595641747944
crude_sol['crudeQuantity[3]'] = 201.29570746971186
crude_sol['crudeQuantity[4]'] = 56.18456149457359
crude_sol['crudeQuantity[8]'] = 162.2466500589715
crude_sol['crudeQuantity[10]'] = 18.848116800048512
crude_y_bound = {
    'crudeQuantity[1]': [0, 201.29570747217807],
    'crudeQuantity[2]': [0, 212.48012718600953],
    'crudeQuantity[3]': [0, 201.29570747217807],
    'crudeQuantity[4]': [0, 199.57869634340224],
    'crudeQuantity[5]': [0, 210.54848966613673],
    'crudeQuantity[6]': [0, 222.1383147853736],
    'crudeQuantity[7]': [0, 196.7885532591415],
    'crudeQuantity[8]': [0, 208.54531001589828],
    'crudeQuantity[9]': [0, 204.3720190779014],
    'crudeQuantity[10]': [0, 210.2623211446741]
}
crude_y_bound = {
    'crudeQuantity[1]': [0, 0],
    'crudeQuantity[2]': [150.5, 151],
    'crudeQuantity[3]': [200, 201.29570747217807],
    'crudeQuantity[4]': [56, 56.5],
    'crudeQuantity[5]': [0, 0],
    'crudeQuantity[6]': [0, 0],
    'crudeQuantity[7]': [0, 0],
    'crudeQuantity[8]': [162, 162.5],
    'crudeQuantity[9]': [0, 0],
    'crudeQuantity[10]': [18.5, 19]
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

options = {
    'max_iter': 1e5,
    'max_time': 3600 * 24,
    'tol': 1e-3,
    'ubd_midpt_fix': 1,
    'ubd_local_solve': 1,
    'ubd_provided': crude_obj,
    'inherit_multiplier': True,
    'aug_lag': False,
    'aug_lag_iter': 3,
    "aug_lag_p":0.01
}

alg = LagrangeanAlgo(m,lag_iter=3, solver='gurobi')
alg.solve(**options)
with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/decomp/results/LG/crude_selection_normal.pkl', 'wb') as f:
    dill.dump(alg, f)