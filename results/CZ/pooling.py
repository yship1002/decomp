from NSPLIB.src.instances.pooling_contract_selection.pooling import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.analyses.convergence_analysis import HausdorffAnalyzer
import dill
def main():
    eps=5
    # create StochasticModel instance
    sto_m = const_model()

    # create CaoZavalaModel instance from sto_m
    m = CaoZavalaModel.from_sto_m(sto_m)
    # build the model
    m.build()

    # declare binary first-stage variables
    binary_ys = ['lambd[1]', 'lambd[2]', 'lambd[3]', 'lambd[4]', 'lambd[5]', 'theta[1]', 'theta[2]', 'theta[3]', 'theta[4]']

    binary_y_val = {y: 0 for y in binary_ys}
    binary_y_val['lambd[1]'] = 1
    binary_y_val['lambd[2]'] = 1
    binary_y_val['lambd[5]'] = 1
    binary_y_val['theta[1]'] = 1
    binary_y_val['theta[4]'] = 1
    m.fix_binary_y(binary_y_val)

    pooling_obj=-1338.2471283376406
    pooling_sol = {
        'A[1]': 300.0, 'A[2]': 201.92127476313524, 'A[3]': 0.0, 'A[4]': 0.0, 'A[5]': 245.18105081826008,
        'S[1]': 247.10232558139526, 'S[2]': 0.0, 'S[3]': 0.0, 'S[4]': 500.0
    }

    pooling_y_bound = {
        'A[1]': [0, 300], 'A[2]': [0, 250], 'A[3]': [0, 0], 'A[4]': [0, 0], 'A[5]': [0, 300],
        'S[1]': [0, 400], 'S[2]': [0, 0], 'S[3]': [0, 0], 'S[4]': [0, 500]
    }
    updated_pooling_y_bound = {
        'A[1]': (300.0-eps, 300.0+eps), 'A[2]': (201.92127476313524-eps, 201.92127476313524+eps), 'A[3]': (0.0, 0.0), 'A[4]': (0.0, 0.0), 'A[5]': (245.18105081826008-eps, 245.18105081826008+eps),
        'S[1]': (247.10232558139526-eps, 247.10232558139526+eps), 'S[2]': (0.0, 0.0), 'S[3]': (0.0, 0.0), 'S[4]': (500.0-eps, 500.0+eps)
    }
    m.update_y_bound(updated_pooling_y_bound)
    alg = CaoZavalaAlgo(m, solver='baron')
    alg.solve(max_iter=1e8, max_time=3600*24, tol=1e-3,ubd_midpt_fix=0,ubd_local_solve=0,ubd_provided=-1338.2471283376406) #JY:relative tolerance
    with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/pooling.pkl', 'wb') as f:
        dill.dump(alg, f)
if __name__ == '__main__':
    main()