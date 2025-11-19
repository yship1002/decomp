from ast import Param

from anyio import value
from pyomo.pyomo.core.base.objective import Objective
from src.models.bb_node import BranchBoundNode, BranchBoundNodeList
from src.models.decomp_model import DecompAlgo
from .cz_model import CaoZavalaModel
from .subgradient_method import SubgradientMethod
from .stepsize_rules import ColorTVRule
from .deflection_rules import STSubgradRule
from pyomo.environ import *
from pyomo.core.expr.visitor import replace_expressions
import logging, logging.config
from pathlib import Path
logging.config.fileConfig(str(Path(__file__).parent.parent)+'/config.ini', disable_existing_loggers=False)
logger_sbb = logging.getLogger('solve.SBBlog')
logger_lbd = logging.getLogger('solve.lowerBound')
# suppress the error message when it attempts to get the objective value from an
# infeasible node
logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)


class LagrangeanModel(CaoZavalaModel):
    """
    The class for solving stochastic programming problems via the idealized
    Lagrangean algorithm.

    aux_models: Contains:
        'lag': dict of lower bounding models, scenario name as key.
        'benders': Benders master problem model.
    """

    def __repr__(self):
        # update this
        s = 'Idealized Lagrangean Stochastic Model'
        if self.name:
            s += f': {self.name}'
        if self.scenarios and self.y_set and self.x_set:
            s += f', {len(self.scenarios)} x {len(self.y_set)} x {len(self.x_set)}'
        return s

    def _build_lag(self):
        """
        Build the Lagrangean subproblems for lower bounding.
        """

        # construct the lower bounding problems via the CZ model
        super()._build_aux_models()

        # rename models
        self.aux_models['lag'] = self.aux_models.pop('lbd')

        # add the `mu * y` term
        # for s in self.scenarios:

        #     # m = self.lagrangean_subproblems[s]
        #     m = self.aux_models['lag'][s]

        #     # add lagrangean multipliers (mu) as parameter
        #     m.mu = Param(self.y_set, initialize={
        #                  idx: 0 for idx in self.y_set}, mutable=True)
            
        #     # don't forget to update consensus_y parameter when starting augmented lagrangean
        #     m.consensus_y=Param(self.y_set, initialize={idx: 0 for idx in self.y_set},mutable=True)
        
            
        #     # delete the old objective to avoid warning
        #     temp_obj=m.obj
        #     m.del_component(m.obj)

        #     # add the (mu * y) term into objective
        #     # if self.obj:
        #     #     def obj(m):
        #     #         return self.obj[s](m, s) + sum(m.mu[i] * m.y[i] for i in self.y_set)
        #     #     m.obj = Objective(expr=obj,sense=minimize)
        #     # else:
        #     #     m.obj = Objective(expr=temp_obj + sum(m.mu[i] * m.y[i] for i in self.y_set),sense=minimize)
        #     m.P=Param(initialize=1.0, mutable=False)
        #     m.obj = Objective(expr=temp_obj + sum(m.mu[i] * (m.y[i]-m.consensus_y[i]) for i in self.y_set)+m.P/2*sum((m.y[i]-m.consensus_y[i])**2 for i in self.y_set),sense=minimize)
    
    def _build_benders(self):
        """
        Build the Benders master problem for lower bounding.
        """

        m = ConcreteModel()
        self.aux_models['benders'] = m

        # variable
        # eta: lower bound of each scenario w.r.t. cuts
        m.eta = Var(self.scenarios, initialize=0)
        # y
        m.y = Var(self.y_set, bounds=self.y_bound, within=self.y_domain)

        # setup constraint list for cuts
        m.lagrangean_cuts = ConstraintList()

        # objective
        def obj_benders_master(m):
            return sum(m.eta[s] for s in self.scenarios)

        m.obj = Objective(rule=obj_benders_master)

    def _build_aux_models(self):
        """Construct lower bounding subproblems.
        """

        self._build_lag()
        self._build_benders()

    def add_cuts(self, mu_set, obj_set):
        """
        Add lagrangean cuts into the Benders master problem.
        args:
            mu_set: List of multipliers.
            obj_set: List of objective values.

        """

        m = self.aux_models['benders']
        if isinstance(mu_set,list): #if there is only one cut
            for i in range(len(mu_set)):
                for s in self.scenarios:
                    m.lagrangean_cuts.add(
                        m.eta[s] >= obj_set[i][s] - sum(m.y[idx] * mu_set[i][s][idx] for idx in self.y_set))
        else:
            for s in self.scenarios:
                    m.lagrangean_cuts.add(
                        m.eta[s] >= obj_set[s] - sum(m.y[idx] * mu_set[s][idx] for idx in self.y_set))

    def clear_cuts(self):
        """
        delete all the lagrangean cuts in the Benders master problem.
        """
        m = self.aux_models['benders']
        m.del_component('lagrangean_cuts_index')
        m.del_component('lagrangean_cuts')
        m.lagrangean_cuts = ConstraintList()
        self.aux_models['benders'] = m

    def fix_binary_y(self, binary_y_val: dict):

        super().fix_binary_y(binary_y_val)

        # regenerate the Benders master problem
        self._build_benders()

    def _fix_aux_model_binary_y(self, binary_y_val: dict):
        # fix these variables in Lagrangean subproblems
        # models = self.aux_models['lbd']
        models = self.aux_models['lag']
        for s in self.scenarios:
            y = models[s].y
            for y_idx in binary_y_val:
                y[y_idx].fix(binary_y_val[y_idx])

    def update_y_bound_aux(self, y_bound):
        """
        update the bound of first stage variable in Lagrangean subproblems and Benders master problem.
        """
        # lagrangean subproblems
        models = self.aux_models['lag']
        for s in models.keys():
            for y_idx in y_bound:
                models[s].y[y_idx].setlb(y_bound[y_idx][0])
                models[s].y[y_idx].setub(y_bound[y_idx][1])
        # benders mater problem
        m = self.aux_models['benders']
        for y_idx in y_bound:
            m.y[y_idx].setlb(y_bound[y_idx][0])
            m.y[y_idx].setub(y_bound[y_idx][1])

class LagrangeanAlgo(DecompAlgo):
    """
    The class for implementing the lower bounding scheme for the Cao & Zavala
    algorithm.

    Reference: https://link.springer.com/article/10.1007/s10898-019-00769-y

    The spatial branch-and-bound algorithm in the class only contains the basic
    bounding techniques from the article. It is not a complete reproduction of
    the full algorithm.

    Attributes:
        lag_iter (int): The Lagrangean iteration time at each node.
        sm (SubgradientMethod): The object to solve the Lagrangean dual problem
        via subgradient method.
    Args:
        model (LagrangeanModel): The Lagrangean model.
        bt_init (bool): Whether to perform bound tightening at the beginning. default False.
        bt_all (bool): Whether to perform bound tightening at each node. default False.
    Raises:
        RuntimeError: If not all binary first-stage variables are fixed.
    """

    def __init__(self, model: LagrangeanModel, bt_init=False, bt_all=False, **kwargs):

        # check if all binary y's have been fixed prior to initialize the
        # algorithm instance, as it affects the sm object

        if not model.binary_y_fixed:
            raise RuntimeError("Not all binary first-stage variables are fixed!")

        super().__init__(model, bt_init=bt_init, bt_all=bt_all, **kwargs)
        
        self._init_sm()

        self.lag_iter = kwargs.get('lag_iter', 200)

    def _init_sm(self, ls=None):
        """
        Initialize the subgradient method object.
        """

        if ls is None:
            ls = self.model.aux_models['lag']

        stepsize_rule = ColorTVRule()
        deflection_rule = STSubgradRule()
        self.sm = SubgradientMethod(ls, self.model.y_set, stepsize_rule, deflection_rule, solver=self.solver)


    def __repr__(self):
        return "Algorithm Instance for " + self.model.__repr__()

    def solve(self, **kwargs):
        """
        Solve the optimization model via SBB. For this model, the Lagrangean
        iteration time can be specified.
        """
        self.lag_iter = kwargs.pop('lag_iter', self.lag_iter)
        super().solve(**kwargs)

    def calc_lbd(self, node: BranchBoundNode, **kwargs):
        """
        Calculate the lower bound for a given node.

        Returns:
            float: The lower bound.
        """

        lbd = 0

        logger_lbd.info(f"Lower bounding the problem at {node.bound}...")
        self.model.update_y_bound_aux(node.bound)
        # before starting, need to initialize param consensus_y based on new bound
        for s in self.model.scenarios:
            # need to update consensus_y based on new bound
            for i in list(node.bound.keys()):
                lb, ub = node.bound[i]
                mid = (lb + ub) / 2
                self.model.aux_models['lag'][s].consensus_y[i]=mid

        # process_sol = {"x1":1727.2601809997955,"x2":16000,"x3":104.23841082714829,"x5":2000}
        # for s in self.model.scenarios:
        #     # need to update consensus_y based on new bound
        #     for i in list(node.bound.keys()):
        #         self.model.aux_models['lag'][s].y[i].fix(process_sol[i])

        # perform augmented lagrangean iteration
        for _ in range(10):
            current_lbd=0.0
            y_scenario={}
            for s in self.model.scenarios:
                m= self.model.aux_models['lag'][s]
                results = self.solver.solve(m, tee=kwargs.get('tee', False),**kwargs)
                penalty_term=value(m.P)/2*sum((value(m.y[i])-value(m.consensus_y[i]))**2 for i in self.model.y_set)
                lag_term=sum(value(m.mu[i]) * (value(m.y[i])-value(m.consensus_y[i])) for i in self.model.y_set)
                current_lbd+=value(results['Problem'][0]['Lower bound'])
                y_scenario[s]={i:value(m.y[i]) for i in self.model.y_set}

            # compute average of y across scenarios
            avg_y = {}
            for i in self.model.y_set:
                avg_y[i] = sum(y_scenario[s][i] for s in self.model.scenarios) / len(self.model.scenarios)

            # update consensus_y in each lag subproblem
            # update mu in each scenario by adding (y - consensus_y)
            for s in self.model.scenarios:
                m = self.model.aux_models['lag'][s]
                for i in self.model.y_set:
                    m.consensus_y[i]=avg_y[i]
                    # compute difference robustly
                    diff = (value(m.y[i]) - value(m.consensus_y[i]))*value(self.model.aux_models['lag'][s].P)
                    m.mu[i] += diff
            print(f"Iteration {_}:")
            self.model.aux_models['lag'][1].mu.pprint()
            a=1
        node.lbd = current_lbd
        return current_lbd

    def _solve_benders(self, node: BranchBoundNode, **kwargs):
        """
        Solve the Benders master problem.
        """

        results = self.solver.solve(self.model.aux_models['benders'], tee=kwargs.get('tee', False),**kwargs)

        # add time
        try:
            self.total_cpu_time += results.solver.time
            node.add_time('benders', results.solver.time)
        except:
            pass

        # store lower bound value & time to node
        node.record_sol(results, 'lbd')

        return value(results['Problem'][0]['Lower bound'])