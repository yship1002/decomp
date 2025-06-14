from NSPLIB.src.instances.poutil.poutil_SP import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
import dill
import numpy as np
def main():
    sto_m = const_model()
    m = CaoZavalaModel.from_sto_m(sto_m)
    m.build()
    alg = CaoZavalaAlgo(m, solver='gurobi')
    #alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=1,ubd_provided=266187.5332404778)
    alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=1)
    with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/DecompConv/data/poutil.pkl', 'wb') as f:
        dill.dump(alg, f)
if __name__ == '__main__':
    main()