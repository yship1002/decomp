# from NSPLIB.src.instances.tanksize.tanksize import const_model
# from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
# from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
# import numpy as np
# import dill
# from multiprocessing import Pool
# y_optimal={'productTankSize[1]': 675.9864410532731,
#  'productTankSize[2]': 541.8555067193988,
#  'productTankSize[3]': 257.58675194535505}
# def generate_new_y_bound(y_optimal,radius,m):
#     new_y_bound={}
#     for i in m.y_bound:
#         left=y_optimal[i]-radius*(m.y_bound[i][1]-m.y_bound[i][0])
#         right=y_optimal[i]+radius*(m.y_bound[i][1]-m.y_bound[i][0])
#         if left<m.y_bound[i][0]:
#             left=m.y_bound[i][0]
#         if right>m.y_bound[i][1]:
#             right=m.y_bound[i][1]
#         new_y_bound[i]=(left,right)
#     return new_y_bound
# def return_alg(radius):
#     sto_m = const_model()
#     m = CaoZavalaModel.from_sto_m(sto_m)
#     m.build()
#     new_y_bound=generate_new_y_bound(y_optimal,radius,m)
#     m.update_y_bound(new_y_bound)
#     alg = CaoZavalaAlgo(m, solver='baron')
#     alg.solve(max_iter=1e5, max_time=3600*8, tol=1e-2,ubd_midpt_fix=1)
#     return alg
# radius_list=np.linspace(0, 0.1, 5)
# radius_list=radius_list[1:]

# with Pool(processes=4) as pool:  # Use 4 worker processes
#         results = pool.map(return_alg, radius_list)
# with open('/storage/coda1/p-jscott319/0/jyang872/DecompConv/m_tanksize.pkl', 'wb') as f:
#     dill.dump(results, f)
from NSPLIB.src.instances.tanksize.tanksize import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.models.lagrangean_model import LagrangeanModel, LagrangeanAlgo
import dill
import numpy as np
from multiprocessing import Pool

def main():
    sto_m = const_model()
    m = CaoZavalaModel.from_sto_m(sto_m)
    m.build()
    alg = CaoZavalaAlgo(m, solver='baron')
    #alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=1,ubd_provided=0.9030394623070541) #JY:relative tolerance
    alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_midpt_fix=1,ubd_local_solve=1)
    with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/DecompConv/data/tanksize.pkl', 'wb') as f:
        dill.dump(alg, f)
if __name__ == '__main__':
    main()
