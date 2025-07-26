from NSPLIB.src.instances.crude_selection.crude_selection import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
import dill
import numpy as np
def main():
    from NSPLIB.src.instances.crude_selection.crude_selection import const_model
    eps=0.5
    sto_m = const_model()
    m = CaoZavalaModel.from_sto_m(sto_m)
    m.build()
    binary_ys = [f'pickCrude[{i}]' for i in range(1, 10 + 1)]
    binary_y_val = {y: 0 for y in binary_ys}
    binary_y_val['pickCrude[2]'] = 1
    binary_y_val['pickCrude[3]'] = 1
    binary_y_val['pickCrude[4]'] = 1
    binary_y_val['pickCrude[8]'] = 1
    binary_y_val['pickCrude[10]'] = 1
    m.fix_binary_y(binary_y_val)
    updated_y_bound=m.y_bound
    updated_y_bound["crudeQuantity[1]"]=(0,0)
    updated_y_bound["crudeQuantity[2]"]=(150.87595641747944-eps,150.87595641747944+eps)
    updated_y_bound["crudeQuantity[3]"]=(201.29570746971186-eps,201.29570746971186+eps)
    updated_y_bound["crudeQuantity[4]"]=(56.18456149457359-eps,56.18456149457359+eps)
    updated_y_bound["crudeQuantity[5]"]=(0,0)
    updated_y_bound["crudeQuantity[6]"]=(0,0)
    updated_y_bound["crudeQuantity[7]"]=(0,0)
    updated_y_bound["crudeQuantity[8]"]=(162.2466500589715-eps,162.2466500589715+eps)
    updated_y_bound["crudeQuantity[9]"]=(0,0)
    updated_y_bound["crudeQuantity[10]"]=(18.848116800048512-eps,18.848116800048512+eps)
    m.update_y_bound(updated_y_bound)
    crude_obj = -18350.146929611754
    alg=CaoZavalaAlgo(m,solver="baron")
    alg.solve(max_iter=1e5, max_time=3600*24, tol=1e-4,ubd_local_solve=1,ubd_midpt_fix=0,ubd_provided=-18350.146929613762)
    with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/crude_selection.pkl', 'wb') as f:
        dill.dump(alg, f)
if __name__ == '__main__':
    main()