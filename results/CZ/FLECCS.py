from NSPLIB.src.instances.FLECCS.src.pyomo_model.model import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
import numpy as np
import dill
from multiprocessing import Pool
from pyomo.environ import SolverFactory
import numpy as np
# def return_alg(radius):
#     n_day = 7
#     week_diff = 52
#     sto_m = const_model(n_day=n_day, week_diff=week_diff)
#     m = CaoZavalaModel.from_sto_m(sto_m)
#     m.build()
#     new_bound={"x_air_adsorb_max":0.7567643358197031,"x_sorbent_total":0.7571021770410511}
#     m.update_y_bound({i: (new_bound[i] - radius,new_bound[i] + radius) for i in new_bound})
#     alg = CaoZavalaAlgo(m, solver='gurobi')
#     alg.solve(max_iter=1e5, max_time=3600*8, tol=1e-3)
#     return alg
# radius_list=np.linspace(0,0.1,5)
# radius_list=radius_list[1:]
# with Pool(processes=4) as pool:  # Use 4 worker processes
#         results = pool.map(return_alg, radius_list)
# with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/DecompConv/m_FLECCS.pkl', 'wb') as f:
#     dill.dump(results, f)
def main():
    n_day = 7
    week_diff = 52
    sto_m = const_model(n_day=n_day, week_diff=week_diff)
    m = CaoZavalaModel.from_sto_m(sto_m)
    m.build()
    alg = CaoZavalaAlgo(m, solver='gurobi')
    #alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=0,ubd_provided=-4.946928843629*10**8) #JY:relative tolerance
    alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=0)
    with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/DecompConv/data/FLECCS.pkl', 'wb') as f:
        dill.dump(alg, f)
if __name__ == '__main__':
    main()
