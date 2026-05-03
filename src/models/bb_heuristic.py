"""
This module contains code to implement the basic version of optimization based bound tightening (OBBT) algorithm.
For every first stage variable, it will call the solver twice to find the lower and upper bounds through setting objective to maximize and minimize the first stage variable
"""
# import src.models.bb_node as bb_node
# import src.models.decomp_model as decomp_model
# from src.utility.solvers import Solver
import logging,logging.config
from pathlib import Path
logging.config.fileConfig(str(Path(__file__).parent.parent)+'/config.ini', disable_existing_loggers=False)
logger_sbb = logging.getLogger('solve.SBBlog')
from pyomo.environ import Objective, value


class bb_heuristic:
    """A naive OBBT algorithm.
    Args:
        model (DecompModel): The decomposition model.
        solver (Solver): The solver to be used.
    """


    def __init__(self, model, solver):

        self.model = model
        self.solver = solver

    def strong_branching(self, node,given_ubd):
        """
        Strong branching on all first stage variables in the node.

        Args:
            node (BranchBoundNode): The node to be tightened.
            given_ubd (float): The given upper bound to compare with.

        """
        logger_sbb.info("Strong branching...")




        old_lbd=node.lbd
        iterator=0

        while iterator<len(node.bound):
            y_idx=list(node.bound.keys())[iterator]
            left_improve=0
            right_improve=0
            left_y_bound = node.bound.copy()
            right_y_bound = node.bound.copy()
            left_y_bound[y_idx] = (node.bound[y_idx][0], (node.bound[y_idx][0]+node.bound[y_idx][1])/2)
            right_y_bound[y_idx] = ((node.bound[y_idx][0]+node.bound[y_idx][1])/2, node.bound[y_idx][1])
            range=(node.bound[y_idx][1]-node.bound[y_idx][0])/2
            # left_improve
            self.model.update_y_bound(left_y_bound)
            for aux_model in self.model.aux_models["lbd"].values():
                results=self.solver.solve(aux_model) # type: ignore
                left_improve += results["problem"][0]['Lower bound']
            self.model.update_y_bound(right_y_bound)
            # right_improve
            for aux_model in self.model.aux_models["lbd"].values():
                results=self.solver.solve(aux_model) # type: ignore
                right_improve += results["problem"][0]['Lower bound']
            left_improve = left_improve-old_lbd
            right_improve = right_improve - old_lbd
            if left_improve<0:
                left_improve=0
            if right_improve<0:
                right_improve=0
            if left_improve == float("inf"):
                if right_improve == float("inf"):
                    raise Exception("Both left and right improve are inf, something is wrong with the model or solver")
                if right_improve<1e-4:
                    node.bound[y_idx]=right_y_bound[y_idx]
                    continue
                else:
                    left_improve=(given_ubd-old_lbd)
                    node.bound[y_idx] = right_y_bound[y_idx]
            if right_improve == float("inf"):
                if left_improve<1e-4:
                    node.bound[y_idx] = left_y_bound[y_idx]
                    continue
                else:
                    left_improve /=range
                    right_improve = (given_ubd-old_lbd)/range
                    node.bound[y_idx] = left_y_bound[y_idx]

            
            self.update_weight(left_improve,right_improve,y_idx,node)
            iterator+=1
        logger_sbb.info("Strong branching ended...")

        return node
    def update_weight(self,left_improve,right_improve,y_idx,node):

        left_improve /= node.bound[y_idx][1] - node.bound[y_idx][0]
        right_improve /= node.bound[y_idx][1] - node.bound[y_idx][0]
        node.weights[y_idx]["left"].append(left_improve)
        node.weights[y_idx]["right"].append(right_improve)

    def get_branching_idx(self,node):
        score=[]
        for y_idx in node.bound:
            left_avg=0
            right_avg=0
            for i in node.weights[y_idx]["left"]:
                if i!=float("inf"):
                    left_avg+=i
            for i in node.weights[y_idx]["right"]:
                if i!=float("inf"):
                    right_avg+=i
            left_avg=left_avg/len(node.weights[y_idx]["left"])* (node.bound[y_idx][1]-node.bound[y_idx][0])
            right_avg=right_avg/len(node.weights[y_idx]["right"])* (node.bound[y_idx][1]-node.bound[y_idx][0])
            if left_avg<right_avg:
                score.append(5.0/6.0*left_avg+1.0/6.0*right_avg)
            else:
                score.append(1.0/6.0*left_avg+5.0/6.0*right_avg)
        # get the index with max average improve
        branching_idx=list(node.bound.keys())[score.index(max(score))]
        return branching_idx