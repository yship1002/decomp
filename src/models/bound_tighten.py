"""
This module contains code to implement the basic version of optimization based bound tightening (OBBT) algorithm.
For every first stage variable, it will call the solver twice to find the lower and upper bounds through setting objective to maximize and minimize the first stage variable
"""
from src.models.bb_node import BranchBoundNode
from src.models.decomp_model import DecompModel
from src.utility.solvers import Solver


from pyomo.environ import Objective, value


class BoundTightenAlgo:
    """A naive OBBT algorithm.
    Args:
        model (DecompModel): The decomposition model.
        solver (Solver): The solver to be used.
    """


    def __init__(self, model: DecompModel, solver: Solver):

        self.model = model
        self.solver = solver
        self.cpu_time = 0

    def tighten_bound(self, node: BranchBoundNode, **kwargs):
        """
        Tighten the bound of all first stage variables in the node.

        Args:
            node (BranchBoundNode): The node to be tightened.

        """
        # logger_sbb.info("Bound tightening...")

        # refresh time
        self.cpu_time = 0

        y_bound = node.bound
        new_y_bound = y_bound.copy()

        # update bound
        self.model.update_y_bound_origin(y_bound)

        m = self.model.origin_model
        # deactivate the original objective
        m.obj.deactivate() # type: ignore

        for y_idx in y_bound:

            # lower bound
            lbd_obj_name = '_obj_' + str(y_idx) + '_lbd'
            setattr(m, lbd_obj_name, Objective(expr=m.y[y_idx], sense=1)) # type: ignore
            # solve the model
            # TODO: consider the tolerance here
            results = self.solver.solve(m, tee=kwargs.get('tee', False), tol=kwargs.get('tol', 1e-4)) # type: ignore
            _obj = getattr(m, lbd_obj_name)
            _lbd = value(_obj)
            # delete the component
            m.del_component(lbd_obj_name) # type: ignore
            # add up CPU time
            self.cpu_time += results.solver.time
            node.add_time('bt', results.solver.time) # type: ignore

            # upper bound
            ubd_obj_name = '_obj_' + str(y_idx) + '_ubd'
            setattr(m, ubd_obj_name, Objective(expr=m.y[y_idx], sense=-1)) # type: ignore
            # solve the model
            # TODO: consider the tolerance here
            results = self.solver.solve(m, tee=kwargs.get('tee', False), tol=kwargs.get('tol', 1e-4)) # type: ignore
            _obj = getattr(m, ubd_obj_name)
            _ubd = value(_obj)
            # delete the component
            m.del_component(ubd_obj_name) # type: ignore
            # add up CPU time
            self.cpu_time += results.solver.time
            node.add_time('bt', results.solver.time) # type: ignore

            # store new bounds
            new_y_bound[y_idx] = [_lbd, _ubd] # type: ignore

        # re-activate the original objective
        m.obj.activate() # type: ignore

        # logger_sbb.info("Done.")

        # record the time and new bound to the node
        node.update_bound(new_y_bound)
