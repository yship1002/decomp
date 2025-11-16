from NSPLIB.src.instances.pooling_contract_selection.pooling import const_model
from src.models.lagrangean_model import LagrangeanAlgo,LagrangeanModel
pooling_obj=-1338.2471283376406
pooling_sol = {
    'A[1]': 300.0, 'A[2]': 201.92127476313524, 'A[3]': 0.0, 'A[4]': 0.0, 'A[5]': 245.18105081826008,
    'S[1]': 247.10232558139526, 'S[2]': 0.0, 'S[3]': 0.0, 'S[4]': 500.0
}

pooling_y_bound = {
    'A[1]': [299,300], 'A[2]': [201,202], 'A[3]': [0, 0], 'A[4]': [0, 0], 'A[5]': [245, 246],
    'S[1]': [247, 248], 'S[2]': [0, 0], 'S[3]': [0, 0], 'S[4]': [499, 500]
}
sto_m = const_model()
m = LagrangeanModel.from_sto_m(sto_m)
m.build()
binary_ys = ['lambd[1]', 'lambd[2]', 'lambd[3]', 'lambd[4]', 'lambd[5]', 'theta[1]', 'theta[2]', 'theta[3]', 'theta[4]']
binary_y_val = {y: 0 for y in binary_ys}
binary_y_val['lambd[1]'] = 1
binary_y_val['lambd[2]'] = 1
binary_y_val['lambd[5]'] = 1
binary_y_val['theta[1]'] = 1
binary_y_val['theta[4]'] = 1
m.fix_binary_y(binary_y_val)
alg = LagrangeanAlgo(m,lag_iter=5, solver='baron')
alg.solve(max_iter=1e5, max_time=3600*24,tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=1,ubd_provided=-1338.2471283376406,inherit_multiplier=True)