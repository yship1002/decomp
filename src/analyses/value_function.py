"""
This module contains code for calculating and visualizing value functions.

Value function refers to the objective value function of two-stage stochastic
programming problems projected to the space of first-stage variables, where
at each fixed point the objective function is minimized w.r.t. the second-stage
variables.

When the dimension of first-stage variable is 1 or 2, the processing step is
straight forward.
When it is more than 2, all but 1/2 first-stage variables need to be fixed to
reduce the dimension to 1/2.
"""

import pickle
import os
from datetime import datetime
import numpy as np
from typing import Optional, Union, Literal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from src.models.cz_model import CaoZavalaModel
from src.utility.solvers import Solver
from src.utility.utility import filter_infty
from src.utility.types import Bound, YIndex, YPoint
import logging
from multiprocessing import Pool

class ValueFunction:
    """
    Class for evaluating and plotting value functions in 1D/2D for a given
    stochastic problem.

    Attributes:
        idx_1 (YIndex): The first chosen y-index for 2D operations.
        idx_2 (YIndex): The second chosen y-index for 2D operations.
        model (CaoZavalaModel): The stored model, with the single-scenario subproblem built.
        total_value_func (numpy.ndarray): The generated total value function values.
        value_func (Dict[Tuple[YIndex, YIndex], numpy.ndarray]): The generated value function values, the two chosen y-indices as key.
        y_grid (Dict[YIndex, np.ndarray]): The y meshgrid in the dict format, the two chosen y-indices as key.
        y_discrete (np.ndarray): The discretized y (1D).
    """

    def __init__(self, model: CaoZavalaModel, **kwargs):
        """

        Args:
            model: The model instance for the problem.
            solver: The solver name. Defaults to 'baron'.
        
        """

        self.model = model
        self.y_mesh = []
        self.y_discrete = []
        self.value_func = {}
        self.total_value_func = []

        self.idx_1 = ''
        self.idx_2 = ''

        # solver
        self.solver = Solver(kwargs.get('solver', 'baron'))

    def __getstate__(self):
        """Return state values to be pickled."""
        return self.model.name, self.y_mesh, self.y_discrete, self.value_func, self.total_value_func, self.idx_1, self.idx_2, None

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        model_name, y_mesh, y_discrete, value_func, total_value_func, idx_1, idx_2, _ = state
        if model_name:
            print(f"model name: {model_name}")
        self.y_mesh, self.y_discrete, self.value_func, self.total_value_func, self.idx_1, self.idx_2 = y_mesh, y_discrete, value_func, total_value_func, idx_1, idx_2

        self.solver = Solver('baron')

    @classmethod
    def load(cls, path='', model: Optional[CaoZavalaModel] = None, solver: Optional[str] = None):
        """
        load an instance from pickle file.
        """

        with open(path, 'rb') as f:
            v_f = pickle.load(f)

        if model:
            v_f.set_model(model)
        if solver:
            v_f.set_solver(solver)

        return v_f

    def set_solver(self, solver: str):
        """
        Helper function to set the solver for the value function calculation.

        Args:
            solver (str): The solver name.
        """
        self.solver = Solver(solver)

    def set_model(self, model: CaoZavalaModel):
        """
        Helper function to set the model for the value function calculation.

        Args:
            model (CaoZavalaModel): The model instance for the problem.

        """
        self.model = model

    def save(self, model_name: str = '', note: str = ''):
        """
        Save the object as a pickle file.

        Args:
            model_name (str): The name of the model. Defaults to ''.
            note (str): The note for the file name. Defaults to ''.
        Returns:
            str: The file name.
        """
        if not model_name:
            if self.model.name:
                model_name = self.model.name
            else:
                raise ValueError("Model name is empty!")

        path = '_results/' + model_name + '/value_func/'
        if not os.path.isdir(path):
            os.makedirs(path)

        file_name = datetime.now().strftime("%m%d%Y_%H%M%S")
        if note:
            file_name += '_' + note
        file_name += '.pickle'

        with open(path + file_name, 'wb') as f:
            pickle.dump(self, f)

        # logger_sbb.info(f"Result saved as {file_name}.")

    def calc_1D(self, idx: YIndex, y_val_fix: YPoint,
                interval: Optional[Bound] = None, step: int = 20):
        """
        Calculate the value function on the given one dimension. When the
        dimension of y is > 1, all the first-stage variables except idx are
        fixed from y_val_fix.

        The range for idx is either given in the arguments or from the bounds
        in the stored model.

        Args:
            idx (YIndex): The first index to discretize on.
            y_val_fix (YPoint): The value of all first-stage variables.
            interval (Bound): The interval of idx for value function evaluation.
            When not provided, the interval in the stored model is used.
            step (int): The number of steps to discretize. Defaults to 20.

        Returns:
            Dict[str, np.ndarray]: The value function for each scenario.
            np.ndarray: The total value function as sum of value functions
        """

        # suppress the error message when it attempts to get the objective value
        # from an infeasible node
        logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)
        # store idx
        self.idx_1 = idx

        # use the y_bound from stored model if not provided
        if interval is None:
            interval = self.model.y_bound[idx] # type: ignore

        # discretize idx in the given interval
        interval_dis = np.linspace(*interval, num=step) # type: ignore
        # store it to the object
        self.y_discrete = interval_dis

        # create dict to store value function for each scenario
        value_func = {s: np.zeros(step) for s in self.model.scenarios}
        total_value_func = np.zeros(step)

        l = len(str(step))
        #print(" " * l * 2 + "  points have been estimated...", end="\r")

        # iterate through each point
        for i in range(step):

            #print(f"{i + 1:>{l}}/{step}", end="\r")

            # update idx value
            y_val_fix[idx] = interval_dis[i]

            # solve all subproblems
            for s in self.model.scenarios:

                m = self.model.aux_models['lbd'][s]

                # update y_val_fix to model
                for y_idx in y_val_fix:
                    m.y[y_idx].fix(y_val_fix[y_idx])

                # solve the model
                results = self.solver.solve(m)

                for y_idx in y_val_fix:
                    m.y[y_idx].unfix()
                value_func[s][i] = filter_infty(results.problem[0]['Upper bound'])

                total_value_func[i] += value_func[s][i]

        # store results
        self.value_func = value_func
        self.total_value_func = total_value_func

        # set the log level back to normal
        logging.getLogger('pyomo.core').setLevel(logging.WARNING)

        return value_func, total_value_func

    def calc_2D(self, idx_1: YIndex, idx_2: YIndex, y_val_fix: YPoint,
                interval_1: Optional[Bound] = None,
                interval_2: Optional[Bound] = None,
                step: int = 5):
        """
        Calculate the value function on the given two dimensions. When the
        dimension of y is > 2, all the first-stage variables except idx_1 and
        idx_2 are fixed.

        The ranges for idx_1 and idx_2 are either given in the arguments or from
        the bounds in the stored model.

        Args:
            idx_1 (YIndex): The first index to discretize on.
            idx_2 (YIndex): The second index to discretize on.
            y_val_fix (YPoint): The value of all first-stage variables.
            interval_1 (Bound): The interval of idx_1 for value function
            evaluation.
            interval_2 (Bound): The interval of idx_2 for value function
            evaluation.
            step (int): The number of steps to discretize. Defaults to 5.
        Returns:
            Dict[str, np.ndarray]: The value function for each scenario.
            np.ndarray: The total value function as sum of value functions
        """

        # suppress the error message when it attempts to get the objective value
        # from an infeasible node
        logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)

        # store indices
        self.idx_1 = idx_1
        self.idx_2 = idx_2

        if interval_1 is None:
            interval_1 = self.model.y_bound[idx_1]
        if interval_2 is None:
            interval_2 = self.model.y_bound[idx_2]

        # construct meshgrid
        y_mesh = np.meshgrid(np.linspace(*interval_1, step), np.linspace(*interval_2, step), indexing='ij') # type: ignore
        self.y_mesh = y_mesh

        # record shape
        shape_0 = y_mesh[0].shape
        shape = y_mesh[0].reshape(-1).shape

        # create dict to store value function for each scenario
        value_func = {s: np.zeros(shape) for s in self.model.scenarios}
        value_func_lower = {s: np.zeros(shape) for s in self.model.scenarios}
        total_value_func = np.zeros(shape)

        size = y_mesh[0].size
        l = len(str(size))

        # iterate through each point
        for i in range(size):

            print(f"{i + 1:>{l}}/{size}", end="\r")

            # update idx_1, idx_2 value
            y_val_fix[idx_1] = y_mesh[0].flatten()[i]
            y_val_fix[idx_2] = y_mesh[1].flatten()[i]

            # solve all subproblems
            for s in self.model.scenarios:

                model = self.model.aux_models['lbd'][s]

                # update y_val_fix to model
                for y_idx in y_val_fix:
                    model.y[y_idx].fix(y_val_fix[y_idx])

                # solve the model
                results = self.solver.solve(model)

                value_func[s][i] = filter_infty(results.problem[0]['Upper bound'])
                value_func_lower[s][i] = filter_infty(results.problem[0]['Lower bound'])
                total_value_func[i] += value_func[s][i]
        #### JY: parallelize the process needed
        # prepare_args = [(y_val_fix,data_idx,idx_1,idx_2,y_mesh,self.model,self.solver) for data_idx in range(size)]
        # with Pool(processes=10) as pool:  # Adjust processes based on CPU cores
        #     results_list = pool.map(self.evaluate_value_func, prepare_args)
        # for s in self.model.scenarios:
        #     value_func[s] = np.array([results[0][s] for results in results_list])
        #     value_func_lower[s] = np.array([results[1][s] for results in results_list])
        # total_value_func += np.array([results[2] for results in results_list])
        #reshape value function list so that they match the meshgrid
        for s in self.model.scenarios:
            value_func[s] = value_func[s].reshape(shape_0)
            value_func_lower[s] = value_func_lower[s].reshape(shape_0)
        total_value_func = total_value_func.reshape(shape_0)

        # store results
        self.value_func = value_func
        self.total_value_func = total_value_func

        # set the log level back to normal
        logging.getLogger('pyomo.core').setLevel(logging.WARNING)

        return value_func, total_value_func,value_func_lower

    def plot_1D(self, s='total'):
        """
        Plot the 1D value function profile as a function of y (first stage variable) straight-forward

        Args:
            s: The scenario for the value function. Defaults to 'total'.
        Returns:
            None
        """

        if self.idx_1 == '':
            raise RuntimeError(f"The value function is not evaluated yet!")

        # use predefined style
        plt.style.use(['./src/utility/' + i + '.mplstyle' for i in ['font-sans', 'size-4-4', 'fontsize']])

        _, ax = plt.subplots()

        if s == 'total':
            value_func = self.total_value_func
        else:
            value_func = self.value_func[s]

        ax.scatter(self.y_discrete, value_func, c='b')
        ax.xaxis.labelpad = 18
        ax.yaxis.labelpad = 18
        ax.set_xlabel(f"{self.idx_1}")
        ax.set_ylabel("value function")

    def plot_2D(self, s: Union[YIndex, Literal['total']] = 'total',
                colorbar_digit=2, x_label_pad=12, y_label_pad=12, x_tick_pad=-5,
                y_tick_pad=1, z_tick_pad=10, x_tick_n=5, y_tick_n=5, z_tick_n=5,
                colorbar_x_loc=0.75):
        """
        Plot the 2D value function profile.

        The two y dimensions are the ones given in the calc_2D method.

        Args:
            s: The scenario for the value function. Defaults to 'total'.
        Returns:
            None
        """

        if self.idx_1 == '' or self.idx_2 == '':
            raise RuntimeError(f"The value function is not evaluated yet!")

        # use predefined style
        plt.style.use(['./src/utility/' + i + '.mplstyle' for i in ['font-sans', 'size-8-8', '3D', 'fontsize-12']])

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        if s == 'total':
            value_func = self.total_value_func
        else:
            value_func = self.value_func[s]

        #surf = ax.plot_surface(self.y_mesh[0], self.y_mesh[1], value_func, cmap=cm.viridis, linewidth=0, antialiased=False) # type: ignore
        surf = ax.plot_surface(self.y_mesh[0], self.y_mesh[1], value_func, cmap=cm.viridis, edgecolor="none")
        ax.set_xlabel(f"{self.idx_1}", loc='right', fontweight='bold')
        ax.set_ylabel(f"{self.idx_2}", fontweight='bold')

        # manually set the pad for z-axis ticks
        for t in ax.xaxis.get_major_ticks(): t.set_pad(x_tick_pad)
        for t in ax.yaxis.get_major_ticks(): t.set_pad(y_tick_pad)
        for t in ax.zaxis.get_major_ticks(): t.set_pad(z_tick_pad)

        # adjust number of axis ticks
        ax.locator_params(nbins=x_tick_n, axis='z')
        ax.locator_params(nbins=y_tick_n, axis='x')
        ax.locator_params(nbins=z_tick_n, axis='y')

        # turn off offset
        ax.ticklabel_format(useOffset=False)

        # set colorbar
        fig.colorbar(surf, shrink=0.4, aspect=20, anchor=(colorbar_x_loc, 0.5), format=FormatStrFormatter(f'%.{colorbar_digit}f'))

        # adjust x, y ticks
        plt.xticks(rotation = 45, verticalalignment='top', horizontalalignment='right')
        plt.yticks(rotation = -15, verticalalignment='center', horizontalalignment='left')

        # adjust x, y label pad
        ax.xaxis.labelpad = x_label_pad
        ax.yaxis.labelpad = y_label_pad
        
        return ax
    def evaluate_value_func(self,arg_tuple):
        y_val_fix,data_idx,idx_1,idx_2,y_mesh,model,solver = arg_tuple[0],arg_tuple[1],arg_tuple[2],arg_tuple[3],arg_tuple[4],arg_tuple[5],arg_tuple[6]
        # update idx_1, idx_2 value
        y_val_fix[idx_1] = y_mesh[0].flatten()[data_idx]
        y_val_fix[idx_2] = y_mesh[1].flatten()[data_idx]
        shape = y_mesh[0].reshape(-1).shape
        value_func = {s: 0 for s in model.scenarios}
        value_func_lower = {s: 0 for s in model.scenarios}
        total_value_func = 0

        # solve all subproblems
        for s in model.scenarios:

            s_model = model.aux_models['lbd'][s]

            # update y_val_fix to model
            for y_idx in y_val_fix:
                s_model.y[y_idx].fix(y_val_fix[y_idx])

            # solve the model
            results = solver.solve(s_model)

            value_func[s] = filter_infty(results.problem[0]['Upper bound'])
            value_func_lower[s] = filter_infty(results.problem[0]['Lower bound'])
            total_value_func += value_func[s]
        return  (value_func, value_func_lower, total_value_func)
