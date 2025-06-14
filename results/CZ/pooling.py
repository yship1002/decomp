from NSPLIB.src.instances.pooling_contract_selection.pooling import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
from src.analyses.convergence_analysis import HausdorffAnalyzer
import dill
def main():
    pooling_obj=-1338.2471283376406
    sto_m = const_model()
    m = CaoZavalaModel.from_sto_m(sto_m)
    m.build()
    crude_sol = {f'crudeQuantity[{i}]': 0. for i in range(1, 10 + 1)}
    crude_sol['crudeQuantity[2]'] = 150.87595641747944
    crude_sol['crudeQuantity[3]'] = 201.29570746971186
    crude_sol['crudeQuantity[4]'] = 56.18456149457359
    crude_sol['crudeQuantity[8]'] = 162.2466500589715
    crude_sol['crudeQuantity[10]'] = 18.848116800048512
    binary_ys = [f'pickCrude[{i}]' for i in range(1, 10 + 1)]
    binary_y_val = {y: 0 for y in binary_ys}
    binary_y_val['pickCrude[2]'] = 1
    binary_y_val['pickCrude[3]'] = 1
    binary_y_val['pickCrude[4]'] = 1
    binary_y_val['pickCrude[8]'] = 1
    binary_y_val['pickCrude[10]'] = 1
    m.fix_binary_y(binary_y_val)
    updated_y_bound=m.y_bound
    updated_y_bound['crudeQuantity[1]']=(0,0)
    updated_y_bound['crudeQuantity[5]']=(0,0)
    updated_y_bound['crudeQuantity[6]']=(0,0)
    updated_y_bound['crudeQuantity[7]']=(0,0)
    updated_y_bound['crudeQuantity[9]']=(0,0)
    m.update_y_bound(updated_y_bound)
    alg=CaoZavalaAlgo(m,solver="baron")
    haus=HausdorffAnalyzer(alg)
    epsilon=abs(0.03*crude_obj)
    new_bound=haus._gen_interval(y=crude_sol, eps=2*epsilon/97.28870987007144*2)  #reuse _gen_interval multiply by 2 to fit our need

    m.update_y_bound(new_bound)
    alg=CaoZavalaAlgo(m,solver="baron")
    alg.solve(max_iter=1e8, max_time=3600*24, tol=0.03,ubd_midpt_fix=0,ubd_local_solve=1,ubd_provided=-18350.146929613762) #JY:relative tolerance
    with open('/storage/home/hcoda1/3/jyang872/p-jscott319-0/DecompConv/L_data/pooling.pkl', 'wb') as f:
        dill.dump(alg, f)
if __name__ == '__main__':
    main()