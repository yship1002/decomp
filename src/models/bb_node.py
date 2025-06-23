"""
This module provides codes to define the class BranchBoundNode and BranchBoundNodeList.

"""
import matplotlib.pyplot as plt
from src.models.subgradient_method import SubgradientMethod
from src.utility.types import YBound, YIndex
from pyomo.opt.results.results_ import SolverResults
from statistics import mean
class BranchBoundNode:
    """Class for single branch-and-bound nodes.

    Attributes:
        bound (dict): The interval bound for the node.
        idx (int): The index of the node in the node list.
        lbd (float): The lower bound of the optimization problem in the node.
        ubd (float): The upper bound of the optimization problem in the node.
        left (BranchBoundNode): The left child node.
        right (BranchBoundNode): The right child node.
        _time (dict): The time used in each calculation step.
        multiplier_set (list): The list of multipliers for the node.
        obj_val_set (list): The list of objective values for the node.
    Args:
        bound (YBound): The interval bound of first stage variables.
    """

    def __init__(self, bound: YBound):

        self.bound = bound.copy()
        self.parent=None
        self.lbd = 0
        self.ubd = 0
        self.left = None
        self.right = None
        self.idx = -1
        self.inner_nodes_count=[]
        self.lbd_time_cz=[]
        self.root_node_time=[]
        self.s_need_solve=[]
        self.save_solver_results=[]
        # time defined as dict for different calculations
        calc_types = ['lbd', 'ubd', 'bt', 'benders', 'lag']
        self._time = {t: 0. for t in calc_types}
        self.lbd_y_optimal=[]
        self.lbd_scenario={}
        self.multiplier_set = []
        self.obj_val_set = []

    def __repr__(self):
        return f"node ({self.lbd}, {self.ubd})"

    def update_bound(self, bound: YBound):
        """Update the interval bound. Used for bound tightening."""
        self.bound = bound.copy()

    def record_sol(self, res: SolverResults, calc_type:str,y_optimal=None,s=None):
        """Record solution for lower/upper bounding."""

        if calc_type == 'lbd':
            self.save_solver_results.append(res)
            # add total lbding time and subproblme lbding time
            try: #this is when subproblem is solved
                self.add_time('lbd', res.problem[0]["Wall time"])
                self.lbd_time_cz.append(res.problem[0]["Wall time"])

            except: # this is when subproblem is inherited from parent
                self.add_time('lbd', res.solver[0]["Wallclock time"]) # type: ignore
                self.lbd_time_cz.append(res.solver[0]["Wallclock time"])

            # collect root node time and add node count (only support baron not gurobi)
            self.root_node_time.append(res.solver.root_node_time)

            try:# when subproblem is solved
                self.add_node_count(int(res["Problem"][0]["Iterations"])) 
            except: # when subproblem is inherited from parent
                self.add_node_count(None)
                
            if 'infeasible' in res.solver.termination_condition:
                self.lbd = float('inf')
                self.lbd_scenario[s]=float('inf')
            else:
                self.lbd_scenario[s]=res.problem[0]['Lower bound'] # type: ignore
                self.lbd = self.lbd + res.problem[0]['Lower bound'] # type: ignore
                self.lbd_y_optimal.append(y_optimal) # type: ignore

        else:  # ubd
            try:
                self.add_time('ubd', res.solver[0]["wall time"]) # type: ignore
            except:
                self.add_time('ubd', res.solver.time)
            if 'infeasible' in res.solver.termination_condition:
                self.ubd = float('inf')
            else:
                self.ubd += res.problem[0]['Upper bound'] # type: ignore
    def add_node_count(self,count:int):
        """Add the count of inner nodes."""
        self.inner_nodes_count.append(count)

    def add_time(self, type: str, time: float):
        """
        Set time for a specific calculation type. For the LG method, when the
        Lagrangean subproblem and Benders master problem are timed, the time is
        also directly added to lower bounding.
        """

        if type in ['l', 'lbd', 'L', 'lower']:
            self._time['lbd'] += time
        elif type in ['u', 'ubd', 'U', 'upper']:
            self._time['ubd'] += time
        elif type in ['benders_master', 'benders', 'bm', 'BM']:
            self._time['benders'] += time
        elif type in ['lagrangean_sub', 'lag', 'ls', 'LS']:
            self._time['lag'] += time
        elif type in ['bt', 'bound_tightening']:
            self._time['bt'] += time
        else:  
            raise ValueError(f"Unidentified time type: {type}")

    @property
    def center(self):
        """Return the center of the first stage variable interval."""

        c = {}
        bound = self.bound
        for y_idx in bound:
            c[y_idx] = (bound[y_idx][0] + bound[y_idx][1]) / 2
        return c

    @property
    def time(self):
        """Return the total time used in the node"""
        return self._time['lbd'] + self._time['ubd'] + self._time['bt']

    def partition(self, idx_p: YIndex):
        """Generate two child nodes via partitioning the bound in the idx_p
        dimension.

        Args:
            idx_p (Any): The y-dimension to partition on.
        """
        
        bound_1 = self.bound.copy()
        bound_2 = self.bound.copy()
        yl, yu = self.bound[idx_p]
        # mid=mean([i[idx_p] for i in self.lbd_y_optimal])
        # if abs(mid-yl)<1e-5 or abs(mid-yu)<1e-5:
        #     mid = (yl + yu) / 2
        # print(f"mid: {mid}, yl: {yl}, yu: {yu}")
        mid = (yl + yu) / 2
        bound_1[idx_p] = [yl, mid]
        bound_2[idx_p] = [mid, yu]

        self.left = BranchBoundNode(bound_1)
        self.left.parent=self
        self.right = BranchBoundNode(bound_2)
        self.right.parent=self
        # pass the cuts
        self.left.multiplier_set = self.multiplier_set.copy()
        self.right.multiplier_set = self.multiplier_set.copy()
        self.left.obj_val_set = self.obj_val_set.copy()
        self.right.obj_val_set = self.obj_val_set.copy()
        a=1
    def store_cuts(self, sm: SubgradientMethod):
        """
        add the multipliers and objective from subgradient method to the node.

        Args:
            sm (SubgradientMethod): The subgradient method instance.
        """
        self.multiplier_set += sm.multiplier_set.copy()
        self.obj_val_set += sm.obj_val_set.copy()

    def set_idx(self, idx: int):
        """
        Set the index of the node.
        Args:
            idx (int): The index to be set.
        """
        self.idx = idx

class BranchBoundNodeList:
    """Class for storing branch-and-bound nodes during solving optimization
    problems.

    Attributes:
        active_nodes (dict): The dict of active nodes.
        node_idx (int): The index of the newest node. Defaults to 0.
        nodes (dict): The dict of nodes.
        root (BranchBoundNode): The root node.
    Args:
        node (BranchBoundNode): The root node.
    """

    def __init__(self, node: BranchBoundNode):

        self.node_idx = 0
        self.root = node
        self.active_nodes = {0: self.root}
        self.nodes = {0: self.root}
        self.fathomed_nodes={"by_inf":[], "by_value":[]}
        # assign index to the root node
        node.set_idx(0)

    def __repr__(self):
        return f"BB node list ({len(self.active_nodes)}/{len(self.nodes)})"

    def add_node(self, node: BranchBoundNode):
        """Add a node to the list. 

        Args:
            node (BranchBoundNode): The node to add to the list.

        Returns:
            int: The index for the added node.
        """

        self.node_idx += 1

        self.active_nodes[self.node_idx] = node
        self.nodes[self.node_idx] = node

        node.set_idx(self.node_idx)

        return self.node_idx

    def delete_node(self, idx: int):
        """Delete the node with the given index from the active node dict.

        Args:
            node_idx (int): The index of the node to be deleted.
        """
        self.active_nodes.pop(idx)
    def fathom_nodes_by_inf(self,idx: int):
        """Fathom nodes by infeasibility.

        Args:
            idx (int): The index of the node to be fathomed.
        """
        self.fathomed_nodes["by_inf"].append(idx)
        self.delete_node(idx) # type: ignore

    def fathom_nodes_by_value_dom(self, ubd: float):
        """Fathom nodes with their lower bounds worse than the given upper
        bound.

        Args:
            ubd (float): The given upper bound.
        """

        to_delete = []

        # go through each active node to check their lower bound
        for idx in self.active_nodes.keys():
            if self.active_nodes[idx].lbd >= ubd: # type: ignore
                to_delete.append(idx)

        for idx in to_delete:
            self.fathomed_nodes["by_value"].append(idx)
            self.delete_node(idx)

    def is_empty(self):
        """check if the active node list is empty.

        Returns:
            bool: check If the active node list is empty.
        """
        return len(self.active_nodes) == 0

    def find_min_lbd(self):
        """Return the node with the minimal lower bound.

        Returns:
            int: The index of the desired node.
            BranchBoundNode: The desired node.
        """
        """
        #TODO: implement binary tree structure in the future to speed up the search
        """
        min_lbd = float("inf")
        min_idx = -1

        for idx, node in self.active_nodes.items():
            if node.lbd < min_lbd: # type: ignore
                min_lbd = node.lbd
                min_idx = idx

        return min_idx, min_lbd

    def get_node(self, idx: int):
        """
        Return the node with the given index.

        Args:
            idx (int): The node index.
        """
        return self.nodes[idx]

    def get_all_bounds(self):
        """Get the bounds stored in the root nodes and all its child nodes.

        Returns:
            list: the list of the interval bound of first stage variables
        """

        nodes = [self.root]
        bounds = []
        while nodes:

            node = nodes.pop(0)
            if node.left:
                nodes.append(node.left)
            if node.right:
                nodes.append(node.right)

            bounds.append(node.bound)

        return bounds

    def get_time(self, type=None):
        """
        Get the total time used in the nodes in this list.
        Args:
            type (str, optional): The type of time to be calculated. Defaults to None.
        Returns:
            float: The total time used in the nodes.
        """

        if type is None:
            key = 'time'
        elif type in ['l', 'lbd', 'L', 'lower']:
            key = 'lbd'
        elif type in ['u', 'ubd', 'U', 'upper']:
            key = 'ubd'
        elif type in ['benders_master', 'benders', 'bm', 'BM']:
            key = 'benders'
        elif type in ['lagrangean_sub', 'lag', 'ls', 'LS']:
            key = 'lag'
        else:  # bound tightening
            key = 'bt'

        nodes = [self.root]
        total_time = 0
        while nodes:

            node = nodes.pop(0)
            if node.left:
                nodes.append(node.left)
            if node.right:
                nodes.append(node.right)

            # total_time += getattr(node, key)
            if not key:
                total_time += node.time
            else:
                total_time += node._time[key]

        return total_time

    def get_distances(self, y_point):
        """
        Get the distances between a given point in first stage and center point in their first stage interval of all nodes in the list.

        Args:
            y_point (dict): a given point in first stage.
        Returns:
            list: the list of distances.
        """
        distances = []

        nodes = [self.root]
        while nodes:

            node = nodes.pop(0)
            if node.left:
                nodes.append(node.left)
            if node.right:
                nodes.append(node.right)

            distances.append(sum([(y_point[y_idx] - node.center[y_idx]) ** 2 for y_idx in y_point])**0.5)

        return distances

    def plot_nodes(self, idx_1: YIndex, idx_2: YIndex):
        """Plot all the nodes visited in 2D in the given dimensions.

        Args:
            idx_1 (Any): The first given dimension.
            idx_2 (Any): The second given dimension.
        """

        plt.style.use(['./src/utility/' + i + '.mplstyle' for i in ['font-sans', 'size-8-8', 'fontsize-12']])
        _, ax = plt.subplots()

        # get bounds of all nodes
        bounds = self.get_all_bounds()

        # get the largest box
        root_node = bounds[0]

        # set up canvas
        x_start, x_end = root_node[idx_1]
        y_start, y_end = root_node[idx_2]
        x_diff = x_end - x_start
        y_diff = y_end - y_start
        ax.set_xlim(x_start - 0.05 * x_diff, x_end + 0.05 * x_diff)
        ax.set_ylim(y_start - 0.05 * y_diff, y_end + 0.05 * y_diff)

        # plot each node as a rectangle
        for bound in bounds:
            x_start, x_end = bound[idx_1]
            y_start, y_end = bound[idx_2]
            rectangle = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, lw=0.5, ec="b", fc="none") # type: ignore
            ax.add_patch(rectangle)

        ax.set_xlabel(f"dimension {idx_1}")
        ax.set_ylabel(f"dimension {idx_2}")

    def count_nodes(self):
        """
        Count the number of nodes in the list.
        Returns:
            int: The number of all nodes in the list.
        """
        return len(self.nodes)

    def count_active_nodes(self):
        """
        Count the number of active nodes in the list.
        Returns:
            int: The number of active nodes in the list.
        """
        return len(self.active_nodes)
