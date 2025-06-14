"""
The module for analyzing the Hausdorff convergence rate of reduced-space spatial branch-and-bound algorithms.

Given a particular algoritm it will calculate distance between minimum of function in an interval centered around a specific point or the global optimizer and the lower bound of the relaxation.

"""
from typing import Optional
import numpy as np
from pyomo.environ import value
from src.utility.types import YBound, YPoint
from src.models.decomp_model import DecompAlgo
from src.models.bb_node import BranchBoundNode
import logging
from pathlib import Path
logging.config.fileConfig(str(Path(__file__).parent.parent)+'/config.ini', disable_existing_loggers=False) # type: ignore
logger = logging.getLogger('solve.convergenceAnalysis')


class HausdorffAnalyzer:
    """
    The class for analyzing Hausdorff convergence rate of reduced-space spatial
    branch-and-bound algorithms.
    """

    def __init__(self, alg: DecompAlgo):
        """
        Args:
            alg (DecompAlgo): The algorithm to be analyzed.
        """
        self.alg = alg

    def analyze(self, y: Optional[YPoint] = None, v: Optional[float] = None,
                eps_min: int = -5, eps_max: int = 0, steps: int = 6,
                y_optimal: bool = False, **kwargs):
        """
        Analyze the Hausdorff convergence rate of the algorithm, either
        around a specific point, or around the global optimizer if y is not
        provided.

        Args:
            y (YPoint, optional): The point where to check the convergence rate. Defaults to None.
            v (float, optional): The global optimal solution. Defaults to None.
            eps_min (int, optional): The minium diameter represented as power of 10. Defaults to -5.
            eps_max (int, optional): The maximum diameter represented as power of 10. Defaults to 0.
            steps (int, optional): The total step number. Defaults to 6.
            y_optimal (bool, optional): If the provided y is optimal. Defaults to False.
            tol (float, optional): The tolerance for the solver. Defaults to 1e-6.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The list of diameters of eps and the list of Hausdorff distances for each eps.
        """

        tol = kwargs.get('tol', 1e-6)

        # if y not provided, solve the problem globally to find the global
        # optimizer
        if y is None:
            logger.info("Optimal solution not provided for the analysis, "
                        "solving the original problem to find optimal y...")
            self.alg.solver.solve(self.alg.model.origin_model) # type: ignore

            # assign the optimizer to y
            y = {}
            for y_idx in self.alg.y_bound:
                y[y_idx] = value(self.alg.model.origin_model.y[y_idx]) # type: ignore

        eps_list = np.logspace(eps_max, eps_min, steps)

        distances = []
        lbds=[]
        logger.info("Calculating the Hausdorff distances...")
        for eps in eps_list:
            distance,lbd= self.calc_distance(y, eps, y_optimal=y_optimal, v=v, tol=tol)
            distances.append(distance)
            lbds.append(lbd)
            logger.info(f"\tdiameter = {eps:.1E}, distance = {distance:.2E}")

        logger.info("Done!")

        return eps_list, distances

    def calc_distance(self, y: YPoint, eps: float,
                      y_optimal: bool = False, v: Optional[float] = None,
                      filter_noise: bool = False, **kwargs) -> float:
        """Calculate the Hausdorff distance at the given interval.

        The newly generated interval is the intersection of the one with the
        specified diameter and the original y bound.

        Args:
            y (YPoint): The specified y point.
            eps (float): The diameter of the interval.
            y_optimal (bool, optional): If the specified point is the global optimizer. Defaults to False.
            v (float, optional): The optimal solution (the minimum of the original function). Defaults to None.
            filter_noise (bool, optional): Whether to reduce small distances (<1e-10) to 0. Defaults to False.
            tol (float, optional): The tolerance for the solver. Defaults to 1e-6.  

        Returns:
            float: The Hausdorff distance.
        """

        tol = kwargs.get('tol', 1e-6)
        alg = self.alg

        # get the interval
        interval = self._gen_interval(y, eps)
  
        # generate node
        node = BranchBoundNode(interval)

        v = self._calc_v(y, interval, y_optimal=y_optimal, v=v,  **kwargs)

        # calculate the minimum of the relaxation (via lower bounding)
        lbd=alg.calc_lbd(node, tol=tol)

        # record Hausdorff metric
        distance = v - lbd

        # filter noise
        if filter_noise and distance < 1e-10:
            print("Filtered noise")
            distance = 0  # special set for slides output

        return distance,lbd

    def _gen_interval(self, y, eps):
        """
        Generate interval with the given center, the diameter, and the original
        bound of the problem.
        Returns:
            YBound: The interval.
        """
        y_bound = {}
        origin_bound = self.alg.y_bound
        for y_idx in y:
            # maximum of lower bound
            __lbd = max(y[y_idx] - 0.5 * eps, origin_bound[y_idx][0])
            # minimum of upper bound
            __ubd = min(y[y_idx] + 0.5 * eps, origin_bound[y_idx][1])
            y_bound[y_idx] = [__lbd, __ubd]
        return y_bound

    def _calc_v(self, y: YPoint, interval: YBound, y_optimal = False, v: Optional[float] = None, **kwargs):
        """
        Calculate the minimum of the value function within the given interval.

        Args:
            y (YPoint): The center of the interval.
            interval (YBound): The interval.
            y_optimal (bool, optional): If the given center point is optimal. Defaults to False.
            v (float, optional): Directly provided v. Defaults to None.
        """
        tol = kwargs.get('tol', 1e-6)

        # return v if directly given
        if v is not None:
            return v

        if y_optimal:
            # solve original model (via upper bounding)
            _tmp_y_bound = {}
            for y_idx in y:
                _tmp_y_bound[y_idx] = [y[y_idx], y[y_idx]]
            _ubd_node = BranchBoundNode(_tmp_y_bound)
            self.alg.calc_ubd(_ubd_node)
            return _ubd_node.ubd
        else:
            # calculate the optimal solution in the given range
            _m = self.alg.model.origin_model

            for y_idx in y:
                _m.y[y_idx].setlb(interval[y_idx][0]) # type: ignore
                _m.y[y_idx].setub(interval[y_idx][1]) # type: ignore

            # globally solve the original model
            results = self.alg.solver.solve(_m, tol=tol) # type: ignore
            if 'infeasible' in results.solver.termination_condition:
                return float('inf')
            else:
                return results.problem[0]['Upper bound']

    # def save_plot(self):
    #     # if not self.format:
    #     #     self.format = 'pdf'
    #     self.format = 'pdf'

    #     if self.format == 'pdf':
    #         self.fig.savefig('imgs/' + self.name + '.pdf', bbox_inches='tight')
    #         print(f"Plot saved as {self.name}.pdf.")
    #     else:  # png
    #         self.fig.savefig('imgs/' + self.name + '.png', dpi=self.spec.dpi,
    #                          bbox_inches='tight')
    #         print(f"Plot saved as {self.name}.png.")

    # def save_data(self):
    #     pickle.dump(self, open('results_data/' + self.name + '.dat', "wb"))
    #     print(f"Data saved as {self.name}.dat.")

    # def replot(self):
    #     # workaround for loading pickled data

    #     # create a dummy figure and use its
    #     # manager to display "fig"
    #     dummy = plt.figure()
    #     new_manager = dummy.canvas.manager
    #     new_manager.canvas.figure = self.fig
    #     self.fig.set_canvas(new_manager.canvas)
    #     self.fig.show()

    # def update_name(self):
    #     self.edited_time = "{:%Y_%m_%d-%H_%M}".format(datetime.now())
    #     self.name = self.base_name + self.edited_time
