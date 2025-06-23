
from src.utility.solvers import Solver
from pyomo.environ import value
from src.utility.types import YBound, PyomoModel, ScenarioIndex, YIndex, Multiplier
from typing import Dict, List, Optional
from .stepsize_rules import StepsizeRule
from .deflection_rules import DeflectionRule


class SubgradientMethod:
    """The class of the subgradient method aiming to solve the Lagrangean dual
    problem.

    This class implements the subgradient method and stores the data for all
    Lagrangean iterations for a single node lower bounding.

    Reference (main): https://link.springer.com/article/10.1007/s12532-017-0120-7
    Reference: https://doi.org/10.1016/j.compchemeng.2012.10.012
    Reference: https://web.stanford.edu/class/ee364b/lectures/subgrad_method_notes.pdf

    NOTE Explanations on the terms:
        Multiplier:
            In the original Lagrangean dual problem, `multiplier` refers to pi.
            After rearranged, it refers to mu.
        Subgradient:
            In the context of Lagrangean dual problem, `subgradient` refers to
            the difference of y (for pi, NOT for mu).
        Updating multiplier process:
            1. get the subgradient (for pi) from the Lagrangean subproblems
            2. update the multiplier pi based on the subgradient
            3. update mu from pi

    Attributes for optimization models:
        models (Dict[ScenarioIndex, PyomoModel]): The Lagrangean subproblems.
        scenarios (list): The set of scenarios.
        y_set (list): The set of y-indices.

    Attributes for storing data:
        direction_set (List[Multiplier]): The list of direction vectors (a
        linear combination of current subgradient and last direction).
        multiplier_set (List[Multiplier]): The list of multipliers.
        obj_val_set (List[Dict[ScenarioIndex, float]]): The list of the
        objective values in each scenario.
        pi_set (List[Multiplier]): The list of un-rearranged multipliers.
        subgradient_set (List[Multiplier]): The list of subgradient.

    Attributes for best cuts:
        best_lbd (float): The best lower bound (the sum of objective values).
        best_mu (Multiplier): The best multiplier.
        best_obj_val (Dict[float]): The best objective values.

    Attributes for overall bounds:
        lbds (List[float]): The lower bounds in each iteration.
        ubd (float): The overall upper bound.

    Attributes for Configurations:
        deflection_rule (DeflectionRule): The deflection rule.
        iter (int): The current iteration number.
        # max_iter (int): The maximum iteration number.
        solver (Solver): The optimization problem solver.
        stepsize_rule (StepsizeRule): The stepsize rule.
    """

    def __init__(self, lag_models: Dict[ScenarioIndex, PyomoModel],
                 y_set: List[YIndex], stepsize_rule: StepsizeRule,
                 deflection_rule: DeflectionRule, solver: Solver, **kwargs):

        # get model info
        self.models = lag_models
        self.scenarios = list(lag_models.keys())
        self.y_set = y_set

        # containers
        self.direction_set = []
        self.pi_set = []
        self.multiplier_set = []
        self.obj_val_set = []
        self.subgradient_set = []

        # bounds
        self.ubd = None
        self.lbds = []

        # best cut
        self.best_obj_val = {s: - float("inf") for s in self.scenarios}
        self.best_mu = {s: {idx: 0 for idx in y_set} for s in self.scenarios}
        self.best_lbd = - float("inf")

        # config
        self.solver = solver
        self.stepsize_rule = stepsize_rule
        self.deflection_rule = deflection_rule
        self.iter = 0
        # self.max_iter = 10

        # timer
        self.total_cpu_time = 0

    def run(self, lag_iter: int, **kwargs):
        """
        Solve the Lagrangean dual problem via the subgradient method.

        NOTE This method is a simplification of the full algorithm scheme,
        mostly because of the standard deflection/projection rule.

        TODO update the step description below
        1. project direction d_{i - 1} (skipped); get stepsize, deflection
        number, direction
        2. check termination criteria
        3. update multiplier
        4. evaluate Lagrangean subproblems
        5. possibly update g_i (skipped)
        args:
            lag_iter (int): The maximum iteration number.
        """

        # reset timer
        self.total_cpu_time = 0

        self._initialize(**kwargs)

        while self.iter < lag_iter:
            # 1. get direction and stepsize
            alpha = self.deflection_rule.deflect()
            direction = self._compute_direction(alpha)
            stepsize = self.stepsize_rule.update_stepsize_factor(direction, self.subgradient_set[-1], self.lbds, self.best_lbd)
            # 2. check stopping criteria
            ...

            # 3. update multiplier
            self._update_multiplier(direction, stepsize)

            # print("iter",self.iter,"stepsize",stepsize,"subgradient",self.subgradient_set[-1])
            # print("direction",direction,"multiplier",self.multiplier_set[-1])
            # 4. solve Lagrangean problems

            self._solve(**kwargs)
            self.iter += 1

        # best_mu=self.best_mu
        # comment out if you want to test perturb
        # perturb=1.001
        # best_mu["s2"][0]*= perturb
        # best_mu["s3"][0]*= perturb
        # best_mu["s1"][0]=-best_mu["s2"][0]-best_mu["s3"][0]
        # self.multiplier_set.append(best_mu)
        # self._set_multiplier(best_mu)
        # self._solve(**kwargs)
    def _initialize(self, **kwargs):

        # set initial multiplier
        init_pi = {s: {idx: 0 for idx in self.y_set} for s in self.scenarios}
        self.pi_set = [init_pi]
        init_mu = {s: {idx: 0 for idx in self.y_set} for s in self.scenarios}
        self.multiplier_set = [init_mu]

        # set multiplier to models
        self._set_multiplier(init_mu)

        # initial objective value
        self.lbds = [- float("inf")]
        self.obj_val_set = []
        self.subgradient_set = []

        # initial direction
        init_direction = {s: {idx: 0 for idx in self.y_set}
                          for s in self.scenarios}
        self.direction_set = [init_direction]

        # reset best lbd
        self.best_lbd = - float("inf")

        # solve Lagrangean subproblems
        self._solve(**kwargs)

        self.iter = 0

    def _solve(self, **kwargs):
        """Solve the Lagrangean subproblems.
        """

        _lbds = {}

        for s in self.scenarios:
            m = self.models[s]
            results = self.solver.solve(m, **kwargs)
            if 'infeasible' in results.solver.termination_condition:
                _lbds[s] = float('inf')
            else:
                _lbds[s] = results['Problem'][0]['Lower bound']
            self.total_cpu_time += results.solver.time

        self._record_subgradient()
        self._record_lbd(_lbds)

    def _compute_direction(self, alpha: float):
        """
        Compute the new direction.

        Args:
            alpha (float): The ratio between the subgradient and the last direction.
        return:
            Multiplier: The new direction.
        """

        subgradient = self.subgradient_set[-1]
        last_direction = self.direction_set[-1]

        new_direction = {}
        for s in self.scenarios:
            new_direction[s] = {}
            for idx in self.y_set:
                new_direction[s][idx] = alpha * subgradient[s][idx] + \
                    (1 - alpha) * last_direction[s][idx]

        self.direction_set.append(new_direction)

        return new_direction

    def _record_lbd(self, lbds):
        """
        Record Lagrangean subproblem objective values and subgradient lower
        bound.
        """

        # record objective values
        scenarios = self.scenarios
        self.obj_val_set.append(lbds)

        # record subgradient lower bound
        subgradient_lbd = sum(self.obj_val_set[-1][s] for s in scenarios)
        self.lbds.append(subgradient_lbd)

        # update best cut
        self._update_best_cut(subgradient_lbd, lbds, self.multiplier_set[-1])

    def _record_subgradient(self):
        """Record the subgradients from the Lagrangean subproblems.
        """

        models = self.models
        scenarios = self.scenarios
        first_scenario = scenarios[0]

        subgradient = {s: {idx: 0 for idx in self.y_set} for s in scenarios}
        for s in scenarios:
            if s == first_scenario:
                pass
            else:
                for y_idx in self.y_set:
                    subgradient[s][y_idx] = value(models[first_scenario].y[y_idx]) - value(models[s].y[y_idx])

        self.subgradient_set.append(subgradient)

    def _set_multiplier(self, mu: Multiplier):
        """Assign the multiplier to Lagrangean subproblems.
        """
        for s in self.scenarios:
            for y_idx in self.y_set:
                self.models[s].mu[y_idx] = mu[s][y_idx]

    def _update_multiplier(self, direction: Multiplier, stepsize: float):
        """Update the multiplier.
        """

        scenarios = self.scenarios

        # update pi
        last_pi = self.pi_set[-1]
        new_pi = {}
        for s in scenarios:
            new_pi[s] = {}
            for y_idx in self.y_set:
                new_pi[s][y_idx] = last_pi[s][y_idx] + \
                    stepsize * direction[s][y_idx]

        self.pi_set.append(new_pi)

        # update mu
        new_mu = {}
        for s in scenarios:
            new_mu[s] = {}
            for y_idx in self.y_set:
                if s == scenarios[0]:
                    new_mu[s][y_idx] = - sum(new_pi[_s][y_idx]
                                             for _s in scenarios[1:])
                else:
                    new_mu[s][y_idx] = new_pi[s][y_idx]
        self.multiplier_set.append(new_mu)

        self._set_multiplier(new_mu)

    def _update_best_cut(self, subgradient_lbd: float, obj_val: Dict[YIndex, float], mu: Multiplier):
        """Update the best cut with a higher subgradient lower bound."""

        if subgradient_lbd > self.best_lbd:
            self.best_lbd = subgradient_lbd
            self.best_obj_val = obj_val
            self.best_mu = mu

    def _upper_bound(self):
        """Calculate the upper bound of the Lagrangean dual problem.

        This step is mainly for estimating the error and terminating the
        algorithm.
        """
        ...