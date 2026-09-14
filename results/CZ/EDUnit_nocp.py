from NSPLIB.src.instances.EDUnits_nocp.EDUnits_nocp import const_model
from src.models.cz_model import CaoZavalaModel, CaoZavalaAlgo
import cProfile
import numpy as np
from pyomo.environ import *
# UBD:57608.7 67.9167 67.8844

sto_m = const_model()
m = CaoZavalaModel.from_sto_m(sto_m)
m.build()
eps=10

EDUnits_sol= {
    'molNStream1': [0.22127814752270333,0.22127814752270333],
    'molWStream1': [123.68614067808144, 123.68614067808144],
    'molNStream2': [0.06147995485275122, 0.06147995485275122],
    'molWStream2': [119.50012464909798, 119.50012464909798],
    'molNStream3': [0.15979819266995188, 0.15979819266995188],
    'molWStream3': [4.186016028983508, 4.186016028983508],
    'molNStream4': [0.1950978428811747, 0.1950978428811747],
    'molWStream4': [5.1568889437620165, 5.1568889437620165],
    'molNStream5': [0.1911678364861698, 0.1911678364861698],
    'molWStream5': [5.010404775321278, 5.010404775321278],
    'molNStream6': [0.003930006395004913, 0.003930006395004913],
    'molWStream6': [0.14648416844074047, 0.14648416844074047],
    'molNStream7': [0.15586818627494714, 0.15586818627494714],
    'molWStream7': [4.0395318605427715, 4.0395318605427715],
    'molNStream8': [0.03529965021122269, 0.03529965021122269],
    'molWStream8': [0.9708729147785081, 0.9708729147785081]
}
#m.update_y_bound(EDUnits_sol)

from pyomo.opt import SolverFactory

# solver=SolverFactory("ipopt")
# solver.solve(m.aux_models["lbd"][3],tee=True)

# solver=SolverFactory("gurobi_direct_minlp",options = {'FeasibilityTol':1e-2})
# solver.solve(m.aux_models["lbd"][3],tee=True)

solver = SolverFactory(
    'baron',
    executable='/Users/jyang872/Desktop/baron/baron',options = {'EpsA': 1e-6,'EpsR':1e-6, 'TDo':0,"AbsConFeasTol":1e-2,"NLPSol":-1,
                                                                "FirstLoc":0,
                                                                'LicName': '/Users/jyang872/Desktop/baron/baronlice.txt'}
)
# solver = SolverFactory(
#     'gurobi_direct_minlp',options = {'FeasibilityTol':1e-2}
# )
#solver.solve(m.aux_models["lbd"][1],tee=True,keepfiles=True)


solver.solve(m.origin_model,tee=True,keepfiles=True)


#alg=CaoZavalaAlgo(m,solver="baron")
#alg.solve(max_iter=1e8, max_time=3600*12, tol=1e-3,ubd_provided=59363)
