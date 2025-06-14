"""
The abstract module for decomposition providing model and algorithm templates.
"""

import pickle
import os
from datetime import datetime
from NSPLIB.src.main import StochasticModel
from src.models.bb_node import BranchBoundNode, BranchBoundNodeList
from src.utility.solvers import Solver
from pyomo.environ import TerminationCondition, Binary, ConcreteModel, value # type: ignore
from time import perf_counter
from abc import ABC, abstractmethod
import logging, logging.config
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
logging.config.fileConfig(str(Path(__file__).parent.parent)+'/config.ini', disable_existing_loggers=False)
logger_ubd = logging.getLogger('solve.upperBound')
logger_sbb = logging.getLogger('solve.SBBlog')
# suppress the error message when it attempts to get the objective value from an
# infeasible node
logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)


class DecompModel(StochasticModel, ABC):
    """
    The abstract class for stochastic programming models solved via
    decomposition.
    The aim of this class is to generate and store auxiliary models for
    bounding.

    Args:
        aux_models (dict): Auxiliary models, to be built by each individual subclass.
        binary_y (list): The list of binary first-stage variable names.
        binary_y_value (dict): The dict of binary first-stage variable values.
        origin_model (PyomoModel): The original Pyomo Model.
        origin_built (bool): If the original Pyomo model has been built.
    """

    def __init__(self):

        super().__init__()

        # auxiliary models
        self.aux_models = {}
        # original model
        self.origin_model = None

        # if the original model has been built
        self.origin_built = False

        # binary first-stage variables
        self.binary_y = []
        # values of binary first-stage variables
        self.binary_y_value = {}

    @classmethod
    def from_sto_m(cls, sto_m: StochasticModel):
        """
        Construct the object with a built StochasticModel instance.

        Args:
            sto_m (StochasticModel): The built StochasticModel instance.
        """

        decomp_m = cls()

        decomp_m.name = sto_m.name
        decomp_m.type = sto_m.type

        decomp_m.scenarios = sto_m.scenarios
        decomp_m.y_set = sto_m.y_set
        decomp_m.x_set = sto_m.x_set

        decomp_m.obj = sto_m.obj
        decomp_m.obj_sense = sto_m.obj_sense

        decomp_m.con_stage_1 = sto_m.con_stage_1
        decomp_m.con_stage_2 = sto_m.con_stage_2

        decomp_m.y_bound = sto_m.y_bound
        decomp_m.x_bound = sto_m.x_bound
        decomp_m.y_domain = sto_m.y_domain
        decomp_m.x_domain = sto_m.x_domain
        decomp_m.y_init = sto_m.y_init
        decomp_m.x_init = sto_m.x_init

        decomp_m.con_stage_1_expr = sto_m.con_stage_1_expr
        decomp_m.con_stage_2_expr = sto_m.con_stage_2_expr
        decomp_m.obj_expr = sto_m.obj_expr

        decomp_m.origin_model = sto_m.pyomo_model
        decomp_m.origin_built = True

        return decomp_m

    def __repr__(self):
        s = 'Stochastic Decomposition Model'
        if self.name:
            s += f': {self.name}'
        if self.scenarios and self.y_set and self.x_set:
            s += f', {len(self.scenarios)} x {len(self.y_set)} x {len(self.x_set)}'
        return s

    def build(self):
        """
        Construct all models for each scenarios.
        """

        #logger_sbb.info("Building models...")

        self._build_aux_models()

        #logger_sbb.info("Done.")

    @abstractmethod
    def _build_aux_models(self):
        pass

    def _identify_binary_y(self):
        """
        Check which first-stage variables are binary.
        """

        y = self.origin_model.y # type: ignore
        for y_name in y:
            if y[y_name].domain == Binary:
                self.binary_y.append(y_name)

    @property
    def binary_y_fixed(self):
        """
        If all the binary y's are fixed.
        """

        if len(self.binary_y) == 0:
            self._identify_binary_y()

        y = self.origin_model.y # type: ignore
        for y_name in self.binary_y:
            if not y[y_name].fixed:
                return False

        return True

    def fix_binary_y(self, binary_y_val: dict):
        """
        Fix the binary first-stage variables.

        For all models (original full-space model and lower bounding ones), all
        binary first-stage variables are fixed.
        These variables are also deleted from y_set and y_bound so that they do
        not affect the SBB procedure.

        Args:
            binary_y_val (dict): The dict of binary variable values.

        Raises:
            KeyError: When not all binary variables are provided in binary_y_val.
        """

        self._identify_binary_y()

        if set(self.binary_y) != set(binary_y_val.keys()):
            raise KeyError("Not values of all binary first-stage variables are provided!")

        self.binary_y_value = binary_y_val

        # update y-set to avoid branching on these binaries
        self.y_set = [y for y in self.y_set if y not in self.binary_y]
        # update y-bound too
        self.y_bound = {k: v for (k, v) in self.y_bound.items() if k not in self.binary_y} # type: ignore

        # fix these variables in original model
        y = self.origin_model.y # type: ignore
        for y_idx in binary_y_val:
            y[y_idx].fix(binary_y_val[y_idx])

        # fix variables in auxiliary models
        self._fix_aux_model_binary_y(binary_y_val)

    @abstractmethod
    def _fix_aux_model_binary_y(self, binary_y_val: dict):
        pass

    def fix_y_origin(self, y_val,tol=0):
        """
        Fix the y value in the original model and auxiliary models. This is done
        by updating the lower and upper bounds to the same value.
        args:
            y_val (dict): The y value to be fixed.
        """

        _y_bound = {}
        for idx in y_val:
            _y_bound[idx] = [y_val[idx]-tol, y_val[idx]+tol]
        self.update_y_bound_origin(_y_bound)
        self.update_y_bound_aux(_y_bound)

    def update_y_bound(self, y_bound):
        """
        Update the y-bound in the original model and auxiliary models.
        args:
            y_bound (dict): The y-bound to be updated.
        """

        self.update_y_bound_origin(y_bound)
        self.update_y_bound_aux(y_bound)

    def update_y_bound_origin(self, y_bound):
        """
        Update the y-bound in the original model.
        """

        model = self.origin_model

        for y_idx in y_bound:
            model.y[y_idx].setlb(y_bound[y_idx][0]) # type: ignore
            model.y[y_idx].setub(y_bound[y_idx][1]) # type: ignore


    @abstractmethod
    def update_y_bound_aux(self, y_bound):
        """
        Update the y-bound in the auxiliary models.
        """
        pass

class DecompAlgo(ABC):
    """
    The general class for decomposition algorithms for nonconvex SP problems.

    Args:
        bt_all (bool): Whether to run bound tightening for all nodes. default false.
        bt_alg (BoundTightenAlgo): The BT algorithm instance.
        bt_init (bool): Whether to run bound tightening at the root node. default false.
        model (DecompModel): The problem model instance.
        iter (int): The iteration number.
        node_idx_min_lbd (int): The B&B node index with the min. lower bound.
        node_list (BranchBoundNodeList): The B&B node list.
        res (SBBResult): The instance for result data.
        solver (Solver): The solver object for the SBB run.
        total_cpu_time (float): The total CPU time for the SBB run.
        y_bound (dict): The y_bound of the root node.
    """

    def __init__(self, model: DecompModel, bt_init=False, bt_all=False, **kwargs):

        self.model = model

        # iteration index
        self.iter = 0

        # node with the min. lbd
        self.node_idx_min_lbd = None

        # total CPU time
        self.total_cpu_time = 0

        # solver
        self.solver = Solver(kwargs.get('solver', 'baron'))

        # whether to run bound tightening at each node
        self.bt_all = bt_all
        # whether to run bound tightening at initialization
        self.bt_init = bt_init
        # BT algorithm instance
        self.bt_alg = None
        if bt_init or bt_all:
            from src.models.bound_tighten import BoundTightenAlgo
            self.bt_alg = BoundTightenAlgo(model, self.solver)

        root = BranchBoundNode(model.y_bound) # type: ignore
        self.node_list = BranchBoundNodeList(root)

        # y_bound, stored for branching
        self.y_bound = self.model.y_bound.copy()

        # instance for result data
        self.res = SBBResult()

    def solve(self, **kwargs):
        """
        Solve the optimization model via SBB.
        """

        # check if all binary y's have been fixed prior to solve
        if not self.model.binary_y_fixed:
            raise RuntimeError("Not all binary first-stage variables are fixed!")

        max_iter = kwargs.get('max_iter', 100)
        max_time = kwargs.get('max_time', 360)
        tol = kwargs.get('tol', 1e-2)
        logger_sbb.info("Solving the problem via SBB...")
        logger_sbb.info(f"\tMaximum iteration: {max_iter}")
        logger_sbb.info(f"\tMaximum time: {max_time}")
        logger_sbb.info(f"\tTolerance: {tol:.0E}")

        # reset CPU time
        self.total_cpu_time = 0

        start_time = perf_counter()
        total_wall_time = 0

        # step 1
        self._initialize(**kwargs)

        logger_sbb.info("Solving the model...")
        logger_sbb.info("=" * 80)
        log_title = "  iteration  "
        log_title += "   CPU time  "
        log_title += "  wall time  "
        log_title += "lower bound  "
        log_title += "upper bound  "
        log_title += "   gap"
        logger_sbb.info(log_title)

        while True:

            if self.node_list.is_empty():
                self.res.status = 'infeasible'
                break

            if self.iter > max_iter:
                self.res.status = 'max_iter_reached'
                break

            if total_wall_time > max_time:
                self.res.status = 'max_time_reached'
                break

            # step 2
            node = self._select_node()

            # step 3
            node_1, node_2 = self._branch(node)

            # step 4
            node_1_lbd,node_2_lbd = self._bound(node_1, node_2, **kwargs) # type: ignore


            # optimality gap
            ubd = self.res.last_ubd
            lbd = self.res.last_lbd
            gap = self.res.get_gap()

            total_wall_time = perf_counter() - start_time

            log = f"{self.iter:>10}   "
            log += f"{self.total_cpu_time:>10.2f}   "
            log += f"{total_wall_time:>10.2f}   "
            log += f"{lbd:>10.4f}   " if abs(lbd) < 1e6 else f"{lbd:>10.4E}   "
            if ubd == float('inf'):
                log += "         -   "
                log += "      -"
            else:
                log += f"{ubd:>10.4f}   " if abs(ubd) < 1e6 else f"{ubd:>10.4E}   "
                log += f"{gap * 100:>6.4f}%"

            logger_sbb.info(log)

            # record time
            self.res.add_cpu_time(self.total_cpu_time)
            self.res.add_wall_time(total_wall_time)

            # count bb nodes
            self.res.add_node_n(self.node_list.count_nodes())
            self.res.add_active_node_n(self.node_list.count_active_nodes())

            if gap < tol:
                self.res.status = 'optimal'
                break

        logger_sbb.info("=" * 80 + "\n\n")

        # record solution
        self.res.record_sol(self.model.origin_model, self.node_list) # type: ignore

        # summary
        self._sol_summary(total_wall_time) # type: ignore

    def _sol_summary(self, total_wall_time):

        if self.res.status == 'optimal':
            logger_sbb.info(f"Optimal solution found.\n")
        elif self.res.status == 'infeasible':
            logger_sbb.info(f"Model is infeasible.\n")
        elif self.res.status == 'max_iter_reached':
            logger_sbb.info(f"Maximum number of iterations reached.\n")
        elif self.res.status == 'max_time_reached':
            logger_sbb.info(f"Maximum solving time reached.\n")
        else:
            ...

        # time spent
        logger_sbb.info(f"total wall time: {total_wall_time:.2f} s")
        logger_sbb.info(f"total CPU time: {self.total_cpu_time:.2f} s")
        # node number
        logger_sbb.info(f"Nodes explored in branch-and-bound: {self.node_list.node_idx + 1}")
        ...

    def _initialize(self, **kwargs):
        """
        Initialize models for the SBB procedure.
        """
        given_ubd = kwargs.get('ubd', float('inf'))

        logger_sbb.info("Initialize the model...")

        self.iter = 0

        root = self.node_list.root

        # bound tightening
        if self.bt_init or self.bt_all:

            logger_sbb.info("Bound tightening...")
            self.bt_alg.tighten_bound(root) # type: ignore
            # update the stored y_bound
            self.y_bound = root.bound.copy()
            self.total_cpu_time += self.bt_alg.cpu_time # type: ignore
            logger_sbb.info("Done.")

        # calculate initial bounds
        self.calc_ubd(root, **kwargs)

        self.calc_lbd(root,**kwargs)
        self.res.add_ubd(min(root.ubd, given_ubd))
        self.res.add_lbd(root.lbd)
        self.res.get_gap()

        # remove node with +inf lbd
        if root.lbd == float('inf'):
            self.node_list.delete_node(0)

        # record the chosen node
        self.node_idx_min_lbd = 0

        # print out the results
        logger_sbb.info(f"\tRoot node lower bound: {root.lbd:.2f}, upper bound: {min(root.ubd, given_ubd):.2f}")
        logger_sbb.info(f"\tTotal initialization CPU time: {root.time:.2f} s")
        logger_sbb.info("Done.")

        # count bb nodes
        self.res.add_node_n(self.node_list.count_nodes())
        self.res.add_active_node_n(self.node_list.count_active_nodes())

    def _select_node(self):

        # select node with lbd of last iteration
        node = self.node_list.active_nodes[self.node_idx_min_lbd] # type: ignore
        # delete node from the node list
        self.node_list.delete_node(self.node_idx_min_lbd) # type: ignore

        # update iteration number
        self.iter += 1

        return node

    def _branch(self, node: BranchBoundNode):
        """
        Branch on the given node.
        """

        bound = node.bound

        width = {}
        # calculate the relative width of each dimension
        for y_idx in self.model.y_set:
            if self.y_bound[y_idx][1] - self.y_bound[y_idx][0] == 0:
                width[y_idx] = 0
            else:
                width[y_idx] = (bound[y_idx][1] - bound[y_idx][0]) / (self.y_bound[y_idx][1] - self.y_bound[y_idx][0])

        # find the dimension with the largest width
        max_idx = max(width, key=width.get) # type: ignore
        # partition on the calculated dimension
        node.partition(max_idx)

        # add child nodes into the list
        self.node_list.add_node(node.left) # type: ignore
        self.node_list.add_node(node.right) # type: ignore

        return node.left, node.right

    def _bound(self, node_1: BranchBoundNode, node_2: BranchBoundNode, **kwargs):
        """
        Calculate lower and upper bounds for both BB nodes (y1 and y2).
        """

        # optional OBBT
        if self.bt_all:
            self.bt_alg.tighten_bound(node_1) # type: ignore
            self.total_cpu_time += self.bt_alg.cpu_time # type: ignore
            self.bt_alg.tighten_bound(node_2) # type: ignore
            self.total_cpu_time += self.bt_alg.cpu_time # type: ignore

        # compute bounds of child nodes
        # y1
        # lower bound
        node_1_lbd=self.calc_lbd(node_1, **kwargs)
        if node_1_lbd == float('inf'):
            self.calc_ubd(node_1, is_lbd_inf=True, **kwargs)
        else:
            self.calc_ubd(node_1, is_lbd_inf=False, **kwargs)
        # y2
        # lower bound
        node_2_lbd=self.calc_lbd(node_2, **kwargs)
        if node_2_lbd == float('inf'):
            self.calc_ubd(node_2, is_lbd_inf=True, **kwargs)
        else:
            self.calc_ubd(node_2, is_lbd_inf=False, **kwargs)

        # remove node with +inf lbd
        for node in [node_1, node_2]:
            if node.lbd == float('inf'):
                self.node_list.fathom_nodes_by_inf(node.idx) # type: ignore

        # get lbd and ubd of this iteration
        # lbd
        self.node_idx_min_lbd, _new_lbd = self.node_list.find_min_lbd()
        self.res.add_lbd(_new_lbd)
        # ubd
        self.res.add_ubd(min(self.res.last_ubd, node_1.ubd, node_2.ubd))

        # fathom nodes
        self.node_list.fathom_nodes_by_value_dom(self.res.last_ubd)
        return node_1_lbd,node_2_lbd


    def calc_ubd(self, node: BranchBoundNode, **kwargs):
        """Generate the upper bound for a given bound by solving the original
        model with y fixed to the midpoint.

        Args:
            node (BranchBoundNode): The node containing the bound.

        Returns:
            float: The upper bound.
        """
        is_lbd_inf=kwargs.get('is_lbd_inf', False) #JY: whether the lbd is already inf
        ubd_provided=kwargs.get('ubd_provided', None) #JY: whether the ubd is provided
        if is_lbd_inf:
            logger_ubd.warning("\tLower bound is infeasible, no need to solve the upper bound.")
            node.ubd=float('inf')
            return float('inf')
        elif ubd_provided is not None:
            logger_ubd.warning("\tupper bound provided no need to solve.")
            node.ubd=ubd_provided
            return ubd_provided

        ubd_midpt_fix=kwargs.get("ubd_midpt_fix",0)  #JY: whether to fix the midpoint of the bound

        ubd_local_solve=kwargs.get("ubd_local_solve",0) #JY: whether to solve the UBD problem locally
        
        y_bound = node.bound

        logger_ubd.info(f"Upper bounding the problem within bound {y_bound}...")

        # specify the midpoint as y_star for the original model
        # generate midpoint
        if ubd_midpt_fix:
            y_midpoint = {}
            for y_idx in y_bound:
                y_midpoint[y_idx] = (y_bound[y_idx][0] + y_bound[y_idx][1]) / 2
            # fix the midpoint to the original model
            self.model.fix_y_origin(y_midpoint)

        else:
            # self.model.update_y_bound_origin(y_bound)
            self.model.update_y_bound(y_bound)

        ubd=0
        results=self.solver.solve(self.model.origin_model, first_loc=ubd_local_solve,tee=kwargs.get('tee', False), tol=kwargs.get('sub_tol', 1e-4)) # type: ignore
        # warn if problem is infeasible
        if 'infeasible' in results.solver.termination_condition:
            logger_ubd.warning("\tSolution is infeasible, value set to infinity.")
            ubd = float('inf')
            node.record_sol(results, 'ubd')
        else:
            ubd += results.problem[0]['Upper bound']
            # record solution (bound and time) to the node
            node.record_sol(results, 'ubd')

        # add up CPU time
        try:
            self.total_cpu_time += results.solver[0].time
        except:
            self.total_cpu_time += results.solver[0]["wall time"]

        logger_ubd.info("\tDone!")

        return ubd

    @abstractmethod
    def calc_lbd(self, node: BranchBoundNode, **kwargs):
        pass

    def save_res(self, model_name: str = '', note: str = ''):

        if not model_name:
            if self.model.name:
                model_name = self.model.name
            else:
                raise ValueError("Model name is empty!")

        self.res.save(model_name, note=note)

    def load_res(self, path: str):

        with open(path, 'rb') as f:
            self.res = pickle.load(f)

        logger_sbb.info(f"Result loaded from {path}.")


class SBBResult:
    """
    The class for storing SBB result data.

    Args:
        abs_gaps (list): The absolute gaps.
        acc_cpu_time (list): The accumulated CPU time for each iteration.
        acc_wall_time (list): The accumulated wall time for each iteration.
        active_bb_n (list): The number of active branch and bound nodes.
        bb_n (list): The total number of branch and bound nodes.
        cpu_time (list): The CPU time for each iteration.
        lbds (list): The lower bounds.
        rel_gaps (list): The relative gaps.
        status (string): The termination status of the SBB run.
        wall_time (list): The wall time for each iteration.
        ubds (list): The upper bounds.
        _sol (dict): The solution (value of first-stage variables and
        objective).
    """

    def __init__(self):

        # bounds of each iteration
        self.ubds = []
        self.lbds = []
        self.rel_gaps = []
        self.abs_gaps = []

        # time
        self.acc_cpu_time = []
        self.acc_wall_time = []
        self.cpu_time = []
        self.wall_time = []

        # algorithm termination status
        self.status = ''

        # BB node number
        self.bb_n = []
        self.active_bb_n = []

        # result
        self._sol = {}

    def add_lbd(self, lbd: float):
        self.lbds.append(lbd)

    def add_ubd(self, ubd: float):
        self.ubds.append(ubd)

    def get_gap(self):
        """
        Get the relative gap of the current iteration. This is only called
        during the SBB run.
        """

        ubd = self.last_ubd
        lbd = self.last_lbd

        abs_gap = ubd - lbd

        try:
            rel_gap = abs_gap / abs(ubd)

        except ZeroDivisionError:
            rel_gap = abs_gap / abs(lbd)

        self.add_rel_gap(rel_gap)
        self.add_abs_gap(abs_gap)
        return rel_gap

    @property
    def last_lbd(self):
        return self.lbds[-1]

    @property
    def last_ubd(self):
        return self.ubds[-1]

    def add_node_n(self, bb_n: int):
        """
        add the number of branch and bound nodes.
        args:
            bb_n (int): The number of branch and bound nodes
        """
        self.bb_n.append(bb_n)

    def add_active_node_n(self, bb_n: int):
        """
        add the number of active branch and bound nodes.
        args:
            bb_n (int): The number of active branch and bound nodes
        """
        self.active_bb_n.append(bb_n)

    def add_rel_gap(self, gap: float):
        """
        add relative gap for each iteration.
        args:
            gap (float): The relative gap for each iteration.
        """
        self.rel_gaps.append(gap)

    def add_abs_gap(self, gap: float):
        """
        add absolute gap for each iteration.
        args:
            gap (float): The absolute gap for each iteration.
        """
        self.abs_gaps.append(gap)

    def add_cpu_time(self, total_time: float):
        """
        add CPU time for each iteration.
        args:
            total_time (float): The total time for each iteration.
        """
        if len(self.acc_cpu_time) > 0:
            self.cpu_time.append(total_time - self.acc_cpu_time[-1])
        else:
            self.cpu_time.append(total_time)

        self.acc_cpu_time.append(total_time)

    def add_wall_time(self, total_time: float):
        """
        add wall time for each iteration.
        args:
            total_time (float): The total time for each iteration.
        
        """
        if len(self.acc_wall_time) > 0:
            self.wall_time.append(total_time - self.acc_wall_time[-1])
        else:
            self.wall_time.append(total_time)

        self.acc_wall_time.append(total_time)

    def record_sol(self, origin_model: ConcreteModel, node_list: BranchBoundNodeList):
        """
        Record final solution.
        args:
            origin_model (ConcreteModel): The original Pyomo model.
            node_list (BranchBoundNodeList): The list of branch and bound nodes.
        raises:
            ValueError: When no feasible solution is found.
        """

        self._sol['y'] = {}
        try:
            for idx in origin_model.y: # type: ignore
                self._sol['y'][idx] = value(origin_model.y[idx]) # type: ignore
        except ValueError:
            logger_sbb.warning('Failed to record solution; no feasible solution from upper bounding.')

        self._sol['final_lbd'] = self.last_lbd
        self._sol['final_ubd'] = self.last_ubd
        self._sol['sbb_node_n'] = len(self.lbds)


        self._sol['time'] = {
            'lbd': node_list.get_time('lbd'),
            'ubd': node_list.get_time('ubd'),
            'bt': node_list.get_time('bt'),
            'total': self.acc_cpu_time[-1]
        }

    @property
    def sol(self):
        return self._sol

    @property
    def total_node_n(self):
        return len(self.lbds)

    def plot_bb_n_gap(self, plot_method='semilogy'):
        """
        Plot the number of nodes against the gap.
        args:
            plot_method (str): The plotting method. Default is 'semilogy'.[semilogy, loglog, plot]
        """
        fathomed_node_n = [i - j for (i, j) in zip(self.bb_n, self.active_bb_n)]
        gap_rec = [1 / g for g in self.abs_gaps]

        plt.style.use(['./src/utility/' + i + '.mplstyle' for i in ['font-sans', 'size-4-4', 'fontsize-12']])

        func_dict = {
            'semilogy': plt.semilogy,
            'loglog': plt.loglog,
            'plot': plt.plot,
        }
        func = func_dict[plot_method]
        lw = 1.75
        func(gap_rec, self.active_bb_n, 'r-', label='active nodes', linewidth=lw)
        func(gap_rec, self.bb_n, 'b-', label='nodes', linewidth=lw)
        func(gap_rec, fathomed_node_n, 'k-', label='fathomed nodes', linewidth=lw)
        plt.xlabel('1 / gap')
        plt.ylabel('node number')
        plt.grid(True, which='major', axis='both')

        plt.legend()

    def save(self, instance_name: str, note: str = ''):
        """
        save the result instance as a pickle file.
        args:
            instance_name (str): The name of the instance.
            note (str): The additional note for the file name.
        """
        path = '_results/' + instance_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)

        file_name = datetime.now().strftime("%m%d%Y_%H%M%S")
        if note:
            file_name += '_' + note
        file_name += '.pickle'

        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)

        logger_sbb.info(f"Result saved as {file_name}.")
