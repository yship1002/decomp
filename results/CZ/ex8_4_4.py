from NSPLIB.src.instances.ex8_4_4.ex8_4_4_SP import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
import numpy as np
import dill
from multiprocessing import Pool
def main():
    sto_m = const_model()
    m = CaoZavalaModel.from_sto_m(sto_m)
    m.build()
    alg = CaoZavalaAlgo(m, solver='baron')
    alg.solve(max_iter=1e8, max_time=3600*24, tol=1e-3,ubd_midpt_fix=0,ubd_local_solve=0,ubd_provided=0.33272286571778786) #JY:relative tolerance
    #alg.solve(max_iter=1e8, max_time=3600*24, tol=1e-3,ubd_midpt_fix=0,ubd_local_solve=0) #JY:relative tolerance
    with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/DecompConv/data/ex8_4_4.pkl', 'wb') as f:
        dill.dump(alg, f)
if __name__ == '__main__':
    main()