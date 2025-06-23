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

        self.solver = SolverFactory('gurobi')

        self.spec['options'] = {
            # allow Gurobi to solve nonconvex MIQP problems
            'NonConvex': 2,
        }

    def solve(self, model: PyomoModel, **kwargs):

        # copy the spec to avoid changing the default settings
        spec = self.spec.copy()

        # overwrite the default settings from imput
        spec['tee'] = kwargs.get('tee', spec['tee'])
        spec['keepfiles'] = kwargs.get('keepfiles', spec['keepfiles'])
        spec['symbolic_solver_labels'] = kwargs.get('symbolic_solver_labels', False)

        # tolerance setting
        spec['options']['MIPGap'] = kwargs.get('tol', 1e-4)

        # save log file
        # spec['logfile'] = './log/gurobi-' + time.strftime("%m-%d-%Y") +'.log'

        # solve the problem
        res = self.solver.solve(model, **spec)

        return res


class BaronSolver(_Solver):

    def __init__(self):
        super().__init__()

        self.solver = SolverFactory('baron')

        self.spec['options'] = {
            # relative optimality gap
            'EpsR': 1e-6,
            # 'EpsA': 1e-1,
            # # maximum iteration
            # 'MaxIter': 1000,
            # # absolute constraint feasible tolerance
            # 'AbsConFeasTol': 1e-6
            # maximum solving time
            'MaxTime': 1000
        }

    def solve(self, model: PyomoModel, **kwargs):

        # NOTE Weird things happen if the baron solver is called multiple times
        # without any options passed to it: the solver can get stuck in an
        # instance for a long time.

        # copy the spec to avoid changing the default settings
        spec = self.spec.copy()

        # overwrite the default settings from imput
        spec['options']["FirstLoc"]=kwargs.get('first_loc', 0)
        spec['tee'] = kwargs.get('tee', False)
        spec['keepfiles'] = kwargs.get('keepfiles', False)
        spec['symbolic_solver_labels'] = kwargs.get('symbolic_solver_labels', False)
        spec['options']['MaxTime'] = kwargs.get('max_time', spec['options']['MaxTime'])

        # tolerance setting
        spec['options']['EpsR'] = kwargs.get('tol', spec['options']['EpsR'])
        # spec['options']['EpsA'] = kwargs.get('tol', spec['options']['EpsA'])

        # save log file
        #spec['logfile'] = './log/baron/' + time.strftime("%H-%M-%S-%m-%d-%Y") +'.log'

        # solve the problem
        res = self.solver.solve(model, **spec)

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
