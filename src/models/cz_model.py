"""
This module contains code to implement the Cao & Zavala algorithm.
"""

from src.models.bb_node import BranchBoundNode
from src.models.decomp_model import DecompAlgo, DecompModel
from pyomo.core.expr.visitor import replace_expressions
from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, TerminationCondition, check_optimal_termination, Constraint
from src.utility.types import YBound
import logging
import pickle
from pathlib import Path
from multiprocessing import Pool
logging.config.fileConfig(str(Path(__file__).parent.parent)+'/config.ini', disable_existing_loggers=False) # type: ignore
logger_lbd = logging.getLogger('solve.lowerBound')
# suppress the error message when it attempts to get the objective value from an
# infeasible node
logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)


class CaoZavalaModel(DecompModel):
    """
    The class for building auxiliary models for the Cao & Zavala algorithm.

    Reference: https://link.springer.com/article/10.1007/s10898-019-00769-y
    Args:
        aux_models: Contains key 'lbd' (dict of lower bounding models, scenario name as key).
    """

    def __repr__(self):
        s = 'Cao & Zavala Stochastic Model'
        if self.name:
            s += f': {self.name}'
        if self.scenarios and self.y_set and self.x_set:
            s += f', {len(self.scenarios)} x {len(self.y_set)} x {len(self.x_set)}'
        return s

    def _build_aux_models(self):
        """Construct lower bounding subproblems.

        These problems only needs to be built once.
        When the SBB node is updated, only the y bounds of these problems need
        to be updated.
        """

        self.aux_models['lbd'] = {}
        models = self.aux_models['lbd']
        _m = self.origin_model

        for s in self.scenarios:

            # This list is created to ensure that the original constraint can be
            # directly passed into subproblems.
            # Second-stage variables (x) are defined by 2 indices (scenarios and
            # x-index), so for each single subproblem a single-scenario set is
            # created.
            # NOTE Is there a way to re-write the functions automatically?
            single_scenario_set = [s]

            m = ConcreteModel()
            models[s] = m

            # declare variables
            m.y = Var(self.y_set, bounds=self.y_bound, within=self.y_domain)
            m.x = Var(single_scenario_set, self.x_set, bounds=self.x_bound, within=self.x_domain)

            # map variable names to the objects
            var_map = {}
            for v in m.component_objects(Var):
                idx_set = v.index_set()
                for v_idx in idx_set:
                    var_map[id(getattr(_m, v.name)[v_idx])] = v[v_idx] # type: ignore

            # objective function
            if self.obj:
                def obj(m):
                    return self.obj[s](m, s)

                m.obj = Objective(expr=obj)
            else:
                new_obj_expr = replace_expressions(self.obj_expr[s], var_map)
                m.obj = Objective(expr=new_obj_expr)

            # g_0
            if self.con_stage_1:
                m.g_0 = ConstraintList()
                for con in self.con_stage_1:
                    m.g_0.add(con(m))
            elif self.con_stage_1_expr:
                for i, con in enumerate(self.con_stage_1_expr):
                    new_con_name = 'g_0_' + str(i)
                    new_con_expr = replace_expressions(con, var_map)
                    setattr(m, new_con_name, Constraint(expr=new_con_expr))

            # g_s
            if self.con_stage_2:
                # add a ConstraintList for each scenario
                if s in self.con_stage_2:
                    # generate attribute from string
                    g_name = 'g_' + str(s)
                    # avoid duplicate name with g0
                    if g_name == 'g_0':
                        g_name = 'g_0_scenario'
                    setattr(m, g_name, ConstraintList())
                    con_list = getattr(m, g_name)
                    for con in self.con_stage_2[s]:
                        con_list.add(con(m, s))
            else:
                if s in self.con_stage_2_expr:
                    for i, con in enumerate(self.con_stage_2_expr[s]):
                        # generate attribute from string
                        g_name = 'g_' + str(s)
                        # avoid duplicate name with g0
                        if g_name == 'g_0':
                            g_name = 'g_0_scenario'
                        new_con_name = g_name + '_' + str(i)
                        new_con_expr = replace_expressions(con, var_map)
                        setattr(m, new_con_name, Constraint(expr=new_con_expr))

    def _fix_aux_model_binary_y(self, binary_y_val: dict):
        models = self.aux_models['lbd']
        for s in self.scenarios:
            y = models[s].y
            for y_idx in binary_y_val:
                y[y_idx].fix(binary_y_val[y_idx])

    def update_y_bound_aux(self, y_bound):

        models = self.aux_models['lbd']
        self.y_bound = y_bound  # JY: add this line to update the y_bound in CZmodel since branchboundnode use this y_bound instead of the one in original_model or aux_models
        for s in models.keys():
            for y_idx in y_bound:
                models[s].y[y_idx].setlb(y_bound[y_idx][0]) # type: ignore
                models[s].y[y_idx].setub(y_bound[y_idx][1]) # type: ignore
                


class CaoZavalaAlgo(DecompAlgo):
    """
    The class for implementing the lower bounding scheme for the Cao & Zavala
    algorithm.

    Reference: https://link.springer.com/article/10.1007/s10898-019-00769-y

    The spatial branch-and-bound algorithm in the class only contains the basic
    bounding techniques from the article. It is not a complete reproduction of
    the full algorithm.
    """

    def __init__(self, model: CaoZavalaModel, bt_init=False, bt_all=False, **kwargs):
        super().__init__(model, bt_init=bt_init, bt_all=bt_all, **kwargs)
    def let_solver_solve(self,arg_list):
        results = self.solver.solve(self.model.aux_models['lbd'][arg_list[0]], tee=arg_list[1], tol=arg_list[2],first_loc=arg_list[3])
        return {"solveresult":results,"y_optimal":{k:v.value for k,v in self.model.aux_models['lbd'][arg_list[0]].y.items()}}
    def calc_lbd(self, node: BranchBoundNode, **kwargs):
        """
        Calculate the lower bound for a given node.

        Returns:
            float: The lower bound.
        """

        lbd = 0

        logger_lbd.info(f"Lower bounding the problem at {node.bound}...")

        # update bound
        self.model.update_y_bound_aux(node.bound)

        s_need_to_solve = []
        if node.parent is not None: # when it is not the root node
            for idx,y_s in enumerate(node.parent.lbd_y_optimal):
                for k,v in y_s.items():
                    try:
                        if not node.bound[k][0] < v < node.bound[k][1]:
                            s_need_to_solve.append(self.model.scenarios[idx])
                            break
                    except KeyError:
                        continue
                   
        else: #when it is the rot node
            s_need_to_solve = self.model.scenarios
        node.s_need_solve=s_need_to_solve

        if len(s_need_to_solve) > 0: #check if there is any scenario that needs to be solved
            # only feed scenario index that needs to be solved
            prepare_args = [(s,kwargs.get('tee', False),kwargs.get('sub_tol', 1e-10),kwargs.get("lbd_local_solve",0)) for s in s_need_to_solve]
            with Pool(processes=len(s_need_to_solve)) as pool:  # Adjust #processes based on CPU cores
                results_list = pool.map(self.let_solver_solve, prepare_args)
        # combine results from no need to solve and need to solve before start recording solutions
        combined_results = {}

        # step 2:formulate a dictionary with index as scenario and results as value
        for s in self.model.scenarios:
            if s in s_need_to_solve:
                #get result from results_list
                combined_results[s] = results_list[s_need_to_solve.index(s)]
            else: # step 1:formulat empty results for scenario that does not need to be solved
                from pyomo.opt.results.results_ import SolverResults
                results = SolverResults()
                print(results['Problem'])
                results.solver.termination_condition=TerminationCondition.optimal
                results['Problem'][0]['Lower bound']=node.parent.lbd_scenario[s]
                results.solver[0]["Wallclock time"]=0
                results.solver.root_node_time=0
                combined_results[s]={"solveresult":results,"y_optimal":node.parent.lbd_y_optimal[s]}


        for scenario,value in combined_results.items():
            results = value["solveresult"]
            logger_lbd.info(f"\tGlobally solve for scenario {scenario}...")

            # width=0
            # for idx,(k,v) in enumerate(node.bound.items()):
            #     width+=(v[1]-v[0])**2
            # # results['Problem'][0]['Lower bound']=0.9529054839994407-kwargs.get('k',1)*width**0.5


            if results.solver.termination_condition == TerminationCondition.infeasible or results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded:
                logger_lbd.warning("\tSolution is infeasible, value set to infinity.")
                lbd = float("inf")

                # terminate the lower bounding in advance
                node.record_sol(results, 'lbd',s=scenario)
                logger_lbd.info(f"\tDone.")
                self.total_cpu_time += results.solver.time
                return lbd
            # optimal
            elif check_optimal_termination(results):
                logger_lbd.info("\tSolution is optimal.")
                lbd += results['Problem'][0]['Lower bound']
                
            # suboptimal
            else:
                # best lower bound
                logger_lbd.info("\tSolution is suboptimal.")
                lbd += results['Problem'][0]['Lower bound']

            node.record_sol(results, 'lbd',y_optimal=value["y_optimal"],s=scenario)

            logger_lbd.info(f"\tDone.")

            # add up CPU time
            try:
                self.total_cpu_time += results.solver.time
            except:
                #self.total_cpu_time += results.solver[0]["wall time"]"Wallclock time"
                self.total_cpu_time += results.solver[0]["Wallclock time"]

        return lbd

    def __repr__(self):
        return "Algorithm Instance for " + self.model.__repr__()
