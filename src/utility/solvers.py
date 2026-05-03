"""
The auxiliary script for providing default solver settings.

Links for Gurobi configuration:
    MIPGap: https://www.gurobi.com/documentation/10.0/refman/mipgap2.html
"""
from abc import ABC, abstractmethod
from pyomo.environ import SolverFactory
import logging
from src.utility.types import PyomoModel
import time
from pathlib import Path

# get pyomo (root) logger
_logger = logging.getLogger('pyomo')
_logger.addHandler(logging.FileHandler(str(Path(__file__).parent.parent)+'/solve.log'))
_logger.setLevel(logging.INFO)


class Solver:
    """
    A wrapper class for solvers.
    """

    def __init__(self, name: str):

        if name.lower() == 'baron':
            self._solver = BaronSolver()
        elif name.lower() == 'scip':
            self._solver = ScipSolver()
        else:  # assume is gurobi
            self._solver = GurobiSolver()

    def solve(self, model: PyomoModel, **kwargs):
        return self._solver.solve(model, **kwargs)


class _Solver(ABC):
    """
    A base class for solvers with predefined specs.
    """

    def __init__(self):

        self.spec = {
            'tee': False,
            'keepfiles': False,
            'options': {}
        }

    @abstractmethod
    def solve(self, model: PyomoModel, **kwargs):
        pass


class GurobiSolver(_Solver):

    def __init__(self):
        super().__init__()
        # from pyomo.contrib.solver.solvers.gurobi_direct_minlp import GurobiDirectMINLP
        self.solver = SolverFactory('gurobi_direct_minlp')

        self.spec['options'] = {
            # allow Gurobi to solve nonconvex MIQP problems
            'NonConvex': 2,
            'OutputFlag': 0,
            "LogToConsole": 0
        }

    def solve(self, model: PyomoModel, **kwargs):

        # copy the spec to avoid changing the default settings
        spec = self.spec.copy()

        # overwrite the default settings from imput
        spec['tee'] = kwargs.get('tee', spec['tee'])
        spec['keepfiles'] = kwargs.get('keepfiles', spec['keepfiles'])


        # tolerance setting
        self.solver.options['MIPGap'] =  1e-15

        # save log file
        # spec['logfile'] = './log/gurobi-' + time.strftime("%m-%d-%Y") +'.log'

        # solve the problem
        res = self.solver.solve(model, **spec)

        return res


class BaronSolver(_Solver):

    def __init__(self):
        super().__init__()

        self.solver = SolverFactory('baron')



    def solve(self, model: PyomoModel, **kwargs):

        # NOTE Weird things happen if the baron solver is called multiple times
        # without any options passed to it: the solver can get stuck in an
        # instance for a long time.

        # copy the spec to avoid changing the default settings
        spec = self.spec.copy()

        # overwrite the default settings from imput
        self.solver.options["FirstLoc"]=kwargs.get('first_loc', 0)
        
        self.solver.options['MaxTime'] = kwargs.get('max_time', 10000)
        self.solver.options["TDo"]=1 # Nonlinear-feasibility-based range reduction option (poor man’s NLPs
        #self.solver.options["MDo"]=0 # marginal based range reduction
        #self.solver.options["LBTTDo"]=0 #linear feasibility based range reducion(poor man's LP)
        #self.solver.options["OBTTDo"]=0 #optimality based range reduction default: 1
        #self.solver.options["PDo"]=0 # probing problems allowed ro redeuce variable bounds default：-2， 
        # self.solver.options["BrVarStra"]=0 # default branching variable strategy by baron
        #self.solver.options["BrPtStra"]=2 # midpoint branch

        self.solver.options["EpsA"]=1E-6
        # save log file
        #spec['logfile'] = './log/baron/' + time.strftime("%H-%M-%S-%m-%d-%Y") +'.log'

        # solve the problem
        res = self.solver.solve(model,tee=kwargs.get('tee', False))

        return res
class ScipSolver(_Solver):

    def __init__(self):
        super().__init__()

        self.solver = SolverFactory('scip')

    def solve(self, model: PyomoModel, **kwargs):

        # solve the problem
        spec = self.spec.copy()
        spec["options"]['heuristics/proximity/mingap']=kwargs.get('tol', 1e-2)
        res = self.solver.solve(model,**spec)

        return res
