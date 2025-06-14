"""
This is the main NSPLIB module. Its main purpose is to provide interfaces
for building a stochastic model (StoModelBuilder) and the data structure
for the model (StochasticModel).

The builder design pattern is utilized for constructing stochastic models
either directly or from an existing Pyomo model. See the link for more
information: https://refactoring.guru/design-patterns/builder
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Dict
import logging
import logging.config
from pyomo.environ import ConcreteModel, Var, ConstraintList, Objective, \
    Constraint, Reals
from pyomo.core.expr.visitor import replace_expressions
from pathlib import Path
logging.config.fileConfig(str(Path(__file__).parent)+'/config.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

PROMPT = {
    'direct': """
To build the model, please provide a dict DICTNAME with the following items and then run director.build(**DICTNAME).
DICTNAME = {
'y_set': # The index set of first-stage variables
'x_set': # The index set of second-stage variables
'scenarios': # The index set of scenarios
'con_1': # The list of constraint functions that only contain first-stage variables
'con_2': # The dict of the rest constraints, scenario name as key, the list of constraint functions as values
'y_bound': # The dict of first-stage variable bounds, variable name as key, [lbd, ubd] as value
'x_bound': # The dict of second-stage variable bounds, (scenario name, variable name) as key, [lbd, ubd] as value
'y_domain': # The dict of first-stage variable domains, variable name as key, Pyomo domain as value
'x_domain': # The dict of second-stage variable domains, (scenario name, variable name) as key, Pyomo domain as value
'y_init': # The dict of first-stage variable domains, variable name as key, variable value as value; or simply pass None
'x_init': # The dict of second-stage variable domains, (scenario name, variable name) as key, variable value as value; or simply pass None
'objs': # The dict of objective functions, scenario name as key, scenario objective function as value
'obj_sense': # Minimize (1) or maximize (-1)
}
    """,
    'pyomo': """
To build the model, please provide a dict DICTNAME with the following items and then run director.build(**DICTNAME).
DICTNAME = {
'pm': # The pyomo model
'y_set': # The index set of first-stage variables
'scenarios': # The index set of scenarios
'con_1': # The list of constraint functions that only contain first-stage variables
'objs': # The dict of objective functions, scenario name as key, scenario objective function as value
'obj_sense': # Minimize (1) or maximize (-1)
}
    """,
}


class StoModelBuilder:
    """
    The director class in the builder design pattern. It is an overall wrapper
    for constructing a stochastic optimization problem via different paths.
    """

    def __init__(self, mode: str, name='', m_type='', hint=True):
        """
        Initialize the object.

        Args:
            mode (str): The building mode, 'direct' or 'pyomo'.
            name (str, optional): The model name. Defaults to ''.
            m_type (str, optional): The model type. Defaults to ''.
            hint (bool, optional): Whether to provide hinting prompt. Defaults
            to True.
        """

        if mode in ['pyomo', 'p']:
            self.mode = 'pyomo'
        elif mode in ['direct', 'd']:
            self.mode = 'direct'
        else:
            raise ValueError(f"Provided mode {mode} not identified!")

        if hint:
            logger.info(PROMPT[self.mode])

        self.name = name
        self.type = m_type

    def build(self, pm=None, y_set=None, x_set=None, scenarios=None, con_1=None,
              con_2=None, y_bound=None, x_bound=None, objs=None, obj_sense=1,
              y_domain=None, x_domain=None, y_init=None, x_init=None):
        """The wrapper building function.

        Returns:
            StochasticModel: The resulting stochastic model.
        """

        # build from existing pyomo model
        if self.mode == 'pyomo':

            builder = _StoModelPyomoBuilder()
            builder.add_meta(self.name, self.type)
            builder.build(pm, y_set, con_1, scenarios, objs, obj_sense)

        # direct build
        else:

            builder = _StoModelDirectBuilder()
            builder.add_meta(self.name, self.type)

            builder.set_sets(scenarios, y_set, x_set)
            builder.set_objs(objs)
            builder.set_con_stage_1(con_1)
            builder.set_con_stage_2(con_2)
            builder.set_y_bound(y_bound, y_domain=y_domain, y_init=y_init)
            builder.set_x_bound(x_bound, x_domain=x_domain, x_init=x_init)
            builder.build()

        # return the resulting model
        return builder.product

    def __repr__(self):
        res = 'Stochastic Model Builder'
        if self.name:
            res += f' for {self.name}'
        return res


class _StoModelBuilder(ABC):
    """
    The abstract builder class providing interface for creating the SP model.

    Args:
        _product (StochasticModel): The resulting SP model (i.e., the builder
        product).
    """

    def __init__(self):
        """
        Initialize the object (initialize _product).
        """
        self.reset()

    def reset(self):
        """
        Reset/Initialize the SP model.
        """
        self._product = StochasticModel()

    @property
    def product(self):
        """
        Output _product and reset it.
        """

        product = self._product
        self.reset()
        return product

    @abstractmethod
    def build(self):
        """
        The abstract method for building the SP model.
        """
        pass

    def add_meta(self, name: str, m_type: str):
        """
        Add meta information to the model.
        """
        self._product.name = name
        self._product.type = m_type

    def set_sets(self, scenarios: List[Any], y_set: List[Any],
                 x_set: List[Any]):
        """
        Set index sets for the model.

        Args:
            scenarios: The scenario set.
            y_set: The index set for y (first-stage variables).
            x_set: The index set for x (second-stage variables).
        """

        self._product.scenarios = list(scenarios)
        self._product.y_set = list(y_set)
        self._product.x_set = list(x_set)

    @abstractmethod
    def set_objs(self):
        """
        Set objectives for each scenario.
        """
        pass

    @abstractmethod
    def set_con_stage_1(self):
        """
        Set constraints that only involve first-stage variables.
        """
        pass

    @abstractmethod
    def set_con_stage_2(self):
        """
        Set constraints that involve second-stage variables.
        """
        pass

    def set_y_bound(self, y_bound: Dict[Any, tuple], y_domain=None, y_init=None):
        """
        Set bounds, domain, and initial values for y (first-stage variables).

        Args:
            y_bound: The dict of y bounds, y index as key, bound tuple (lb, ub)
            as value.
            y_domain: The dict of y domains, y index as key, domain set as
            value. Defaults to None.
            y_init: The dict of y initial values, y index as key, init value as
            value. Defaults to None.
        """

        _check_set(self._product.y_set, y_bound, 'y bound')
        self._product.y_bound = y_bound

        # when not provided, all variables are assumed to be continuous
        if not y_domain:
            y_domain = {}
            for y_idx in self._product.y_set:
                y_domain[y_idx] = Reals
        self._product.y_domain = y_domain

        # None can be assigned as initial values in Pyomo
        self._product.y_init = y_init

    def set_x_bound(self, x_bound: Dict[tuple, tuple], x_domain=None, x_init=None):
        """
        Set bounds, domain, and initial values for x (second-stage variables).

        Args:
            X: The dict of x bounds, (scenario, x index) as key, bound
            tuple (lb, ub) as value.
            x_domain: The dict of x domains, scenario as outer key, x index as
            inner key, domain set as value (Reals/Binary). Defaults to None.
            y_init: The dict of x initial values, scenario as outer key, x index
            as inner key, init value as value. Defaults to None.
        """

        self._product.x_bound = x_bound

        # when not provided, all variables are assumed to be continuous
        if not x_domain:
            x_domain = {}
            for s in x_bound:
                x_domain[s] = Reals
        self._product.x_domain = x_domain

        # None can be assigned as initial values in Pyomo
        self._product.x_init = x_init


class _StoModelDirectBuilder(_StoModelBuilder):
    """
    The class for building SP problems directly.
    """

    def build(self):
        """
        Build pyomo model.

        All individual objective functions are weighted by 1.
        """

        sp = self._product
        pyo_model = self._product.pyomo_model

        # declare variables
        pyo_model.y = Var(sp.y_set, bounds=sp.y_bound, within=sp.y_domain)
        pyo_model.x = Var(sp.scenarios, sp.x_set,
                          bounds=sp.x_bound, within=sp.x_domain)

        # objective function
        # if sp.obj_sense == -1:
        #     _sign = -1
        # else:
        #     _sign = 1
        def obj(_m):
            return sp.obj_sense * sum([sp.obj[s](_m, s) for s in sp.scenarios])
        pyo_model.obj = Objective(expr=obj)

        # g_0
        if sp.con_stage_1:
            # TODO CONSIDER changing it to Constraint() to be consistent with
            # build_from_pyomo
            pyo_model.g_0 = ConstraintList()
            for g_0 in sp.con_stage_1:
                pyo_model.g_0.add(g_0(pyo_model))

        # g_s
        if sp.con_stage_2:
            # add a ConstraintList for each scenario
            for scenario in sp.scenarios:
                if scenario in sp.con_stage_2:
                    # generate attribute from string
                    g_name = 'g_' + str(scenario)
                    # avoid duplicate name with g0
                    if g_name == 'g_0':
                        g_name = 'g_0_scenario'
                    setattr(pyo_model, g_name, ConstraintList())
                    con_list = getattr(pyo_model, g_name)
                    for con in sp.con_stage_2[scenario]:
                        con_list.add(con(pyo_model, scenario))

    def set_objs(self, f_s: Dict[Any, Callable], obj_sense=1):
        """
        Set objective functions for each scenario.

        Args:
            f_s: The dict of objective functions, scenario as key, objective
            function as value.
            obj_sense: The optimization direction (1: minimize, -1: maximize).
        """

        _check_set(self._product.scenarios, f_s, 'objective function')
        self._product.obj_sense = obj_sense
        self._product.obj = f_s

    def set_con_stage_1(self, g_0: List[Callable]):
        """
        Set constraints that only contain first-stage variables.

        Args:
            g_0: The list that contains functions of the constraints to be
            included.
        """
        self._product.con_stage_1 = g_0

    def set_con_stage_2(self, g_s: Dict[Any, List[Callable]]):
        """
        Set constraints that contain second-stage variables.

        Args:
            g_s: The dict that contains functions of the constraints to be
            included, scenario as key, list of constraint functions as value.
        """
        _check_set(self._product.scenarios, g_s, 'second stage constraint')
        self._product.con_stage_2 = g_s

    def __repr__(self):
        return 'Stochastic Model Direct Builder'


class _StoModelPyomoBuilder(_StoModelBuilder):
    """
    The class for building SP problems via built Pyomo models.
    """

    def build(self, pm: ConcreteModel, var1_names: List[str],
              con1_names: List[str], scenarios: List[Any],
              objs: Dict[Any, Callable], obj_sense: int = 1):
        """
        Build model from a constructed Pyomo model.

        Args:
            var1_names: The list of names of fist-stage variables.
            con1_names: The list of names of constraints only containing
            first-stage variables.
            obj_sense: minimize (1) or maximize (-1).
            objs: The dict of objective functions.
            pm: The model to be imported.
            scenarios: The list of scenarios.
        """

        sp = self._product

        # 1. categorize variables
        old_var1 = []
        old_var2 = []
        for var in pm.component_objects(Var, active=True):
            old_var1.append(
                var) if var.name in var1_names else old_var2.append(var)

        # 2. categorize constraints
        old_con1 = []
        old_con2 = []
        # NOTE the following line go through all constraints, including
        # Constraint and ConstraintList
        for con in pm.component_objects(Constraint, active=True):
            old_con1.append(
                con) if con.name in con1_names else old_con2.append(con)

        # 3. flatten and rename variables
        old_var1_f = _flatten(old_var1)
        old_var2_f = _flatten(old_var2)
        var2_name_map = _rename_ss(old_var2)

        # 4. construct variable bounds
        y_bound, y_domain, y_init = _get_y_config(old_var1_f)
        x_bound, x_domain, x_init = _get_x_config(old_var2_f, var2_name_map)

        # 5. get y- and x-indices
        y_set = list(y_bound.keys())
        x_set = [j for (i, j) in x_bound.keys() if i == scenarios[0]]

        # 6. rename constraints
        old_con1_f = _flatten(old_con1)
        old_con2_f = _flatten(old_con2)
        con2_name_map = _rename_ss(old_con2)

        old_con2_nested = _get_con2_map(old_con2_f, con2_name_map)

        # 7. fetch model
        _model = sp.pyomo_model

        # 8. create variables for the new model
        _model.y = Var(y_set, bounds=y_bound,
                       within=y_domain, initialize=y_init)
        _model.x = Var(scenarios, x_set, bounds=x_bound,
                       within=x_domain, initialize=x_init)

        # 9. generate variable map between two models
        var_map = _get_var1_map(old_var1, _model)
        var_map.update(_get_var2_map(old_var2, _model))

        # 10. add new constraints
        sp.con_stage_2_expr = {}
        for scenario in scenarios:
            sp.con_stage_2_expr[scenario] = []
            if scenario in old_con2_nested:
                # generate attribute from string
                g_name = 'g_' + str(scenario)
                # avoid duplicate name with g0
                if g_name == 'g_0':
                    g_name = 'g_0_scenario'

                for con_name in old_con2_nested[scenario]:
                    new_con_name = g_name + '_' + con_name
                    con = old_con2_nested[scenario][con_name]
                    new_expr = replace_expressions(con.expr, var_map)
                    setattr(_model, new_con_name, Constraint(expr=new_expr))
                    # add expression to attribute
                    sp.con_stage_2_expr[scenario].append(new_expr)

        sp.con_stage_1_expr = []
        g_name = 'g_0'
        for con in old_con1_f:
            new_con_name = g_name + '_' + con.name
            new_expr = replace_expressions(con.expr, var_map)
            setattr(_model, new_con_name, Constraint(expr=new_expr))
            # add expression to attribute
            sp.con_stage_1_expr.append(new_expr)

        # 11. add objectives
        obj_expr = {}
        for scenario in objs:

            # 11.1. warp up the sub objectives
            if obj_sense == -1:
                def sub_obj(model):
                    return - objs[scenario](model, scenario)
            else:
                def sub_obj(model):
                    return objs[scenario](model, scenario)

            # 11.2. set objective (with the old variables) to the old model
            obj_name = 'obj_' + str(scenario)
            setattr(pm, obj_name, Objective(expr=sub_obj))
            _obj = getattr(pm, obj_name)

            # 11.3. store the updated expressions
            obj_expr[scenario] = replace_expressions(_obj, var_map)

        # 11.4. sum up the expressions
        overall_obj = sum(v for v in obj_expr.values())

        # 11.5. assign the overall obj to the model
        _model.obj = Objective(expr=overall_obj)

        # 11.6. remove extra objectives from original model
        for scenario in objs:
            pm.del_component('obj_' + str(scenario))

        # 12. set arguments
        self.set_sets(scenarios, y_set, x_set)
        self.set_y_bound(y_bound, y_domain=y_domain, y_init=y_init)
        self.set_x_bound(x_bound, x_domain=x_domain, x_init=x_init)
        self.set_objs(obj_expr)

    def __repr__(self):
        return 'Stochastic Model Pyomo Builder'

    def set_objs(self, f_s: Dict[Any, Callable], obj_sense=1):
        """
        Set objective functions for each scenario.

        Args:
            f_s: The dict of objective functions or expressions, scenario
            as key, objective function as value.
            obj_sense: The optimization direction (1: minimize, -1: maximize).
        """

        _check_set(self._product.scenarios, f_s, 'objective function')

        self._product.obj_sense = obj_sense
        self._product.obj_expr = f_s

    def set_con_stage_1(self, g_0: List[Callable]):
        """
        Set constraints that only contain first-stage variables.

        Args:
            g_0: The list that contains functions or expressions of the
            constraints to be included.
        """
        self._product.con_stage_1_expr = g_0

    def set_con_stage_2(self, g_s: Dict[Any, List[Callable]]):
        """
        Set constraints that contain second-stage variables.

        Args:
            g_s: The dict that contains functions or expressions of the
            constraints to be included, scenario as key, list of constraint
            functions or expressions as value.
        """
        _check_set(self._product.scenarios, g_s, 'second stage constraint')
        self.con_stage_2_expr = g_s


class StochasticModel:
    """
    The class for two-stage stochastic programming problems. This is the product
    class within the builder pattern.

    TODO: complete unittest

    Args:
        con_stage_1 (list): The list of constraints only containing first-stage
        variables.
        con_stage_1_expr (list): The list of constraint expressions only
        containing first-stage variables.
        con_stage_2 (dict): The nested dict of constraints containing
        second-stage variables, scenario as outer key, constraint name as inner
        con_stage_2_expr (dict): The nested dict of constraint expressions
        containing second-stage variables, scenario as outer key, constraint
        name as inner key, constraint function as value.
        name (str): The problem name.
        obj (dict): The dict of objective functions for each scenario.
        obj_expr (dict): The dict of objective function expressions for each
        scenario.
        obj_sense (int): The optimization direction (1: minimize, -1: maximize).
        pyomo_model (ConcreteModel): The built Pyomo model.
        scenarios (list): The list of scenarios for the second stage.
        type (str): The problem type.
        x_bound (dict): The nested dict for bounds of second-stage variables,
        scenario as outer key, x-index as inner key, bound tuple (lb, ub) as
        value.
        x_domain (dict): The nested dict for domains of second-stage variables,
        scenario as outer key, x-index as inner key, domain set (Reals/Binary)
        as value.
        x_init (dict): The initial values of second-stage variables, default to
        None.
        x_set (list): The list of second-stage variable indices.
        y_bound (dict): The dict for bounds of first-stage variables, y-index
        as key, bound tuple (lb, ub) as value.
        y_domain (dict): The dict for domains of first-stage variables,
        y-index as inner key, domain set (Reals/Binary) as value.
        y_init (dict): The initial values of first-stage variables, default to
        None.
        y_set (list): The list of first-stage variable indices.
    """

    def __init__(self):
        """Initialize an instance.
        """

        self.name = ''
        self.type = ''

        self.scenarios = []
        self.y_set = []
        self.x_set = []

        self.obj = {}
        self.obj_sense = 1

        self.con_stage_1 = []
        self.con_stage_2 = {}

        self.y_bound = {}
        self.x_bound = {}
        self.y_domain = {}
        self.x_domain = {}
        self.y_init = None
        self.x_init = None

        self.pyomo_model = ConcreteModel()

        self.con_stage_1_expr = []
        self.con_stage_2_expr = {}
        self.obj_expr = {}

    def __repr__(self):
        msg = 'Stochastic Model'
        if self.name:
            msg += f": {self.name}"
        if self.scenarios and self.y_set and self.x_set:
            msg += f', {len(self.scenarios)} x {len(self.y_set)} x {len(self.x_set)}'
        return msg


def _check_set(ori_set: List[Any], new_dict: dict, dict_name: str = ''):
    """
    Check if all indices in ori_set are covered in new_dict.

    Args:
        ori_set: The list that contains all indices.
        new_dict: The dict to be checked.
        dict_name: The name of the dict. Defaults to ''.

    Raises:
        IndexError: Raised when there are extra indices in new_dict.
    """

    extra_idx = [i for i in new_dict if i not in ori_set]

    if extra_idx:
        message = "The"
        if dict_name:
            message += f" {dict_name}"
        message += f" dict has extra indices: {extra_idx}\n"
        raise IndexError(message)

    missing_idx = [i for i in ori_set if i not in new_dict]
    if missing_idx:
        message = "WARNING: The following indices are missing"
        if dict_name:
            message += f" from the {dict_name} dict"
        message += f": {missing_idx}\n"
        logger.warning(message)

    return


def _flatten(old_list: list):
    """
    Flatten the list containing Pyomo indexed objects.

    This works for both variables and constraints. The indexed objects will be
    replaced with flattened, scalar objects.

    Args:
        old_list: The list to be flattened.

    Returns:
        list: The flattened object list.
    """

    new_list = []

    for obj in old_list:
        idx_set = obj.index_set()
        for idx in idx_set:
            try:
                new_list.append(obj[idx])
            # NOTE: workaround for Constraint.Skip
            except KeyError:
                pass

    return new_list


def _rename_ss(ss_list: list):
    """
    Rename second-stage objects (variables and scenarios).

    As the second-stage variables are not indexed by scenarios, their names are
    changed (i.e., the scenario index is removed). In the result those variables
    are also flattened (indexed variables are replaced with several scalar
    variables). The same for the constraints.

    NOTE This step requires that the scenario is the last index for the
    variables in the original pyomo model.

    Args:
        ss_list: The list to be renamed with the original variable data.

    Returns:
        dict: The renamed variable dict, with the original (scalar) variable
        name as key, the updated (scalar) variable data as value.
    """

    name_dict = {}

    # NOTE name: name of indexed variable, e.g., alpha
    #      _name: name of scalar variable, e.g., alpha[1, 'S1']
    for obj in ss_list:

        idx_set = obj.index_set()

        for idx in idx_set:

            try:
                _name = obj[idx].name

                # multi-indexed
                if isinstance(idx, tuple):
                    _name_new = ','.join(_name.split(',')[:-1]) + ']'
                    name_dict[_name] = _name_new

                # single-indexed (only by scenario)
                else:
                    # can directly use the indexed variable name
                    name_dict[_name] = obj.name
            # NOTE: workaround for Constraint.Skip
            except KeyError:
                pass

    return name_dict


def _get_y_config(var_list: list):
    """
    Get bounds, domains, and initial values for first-stage variables.

    Args:
        var_list: The flattened variable list.

    Returns:
        tuple(dict, dict, dict): The variable bound dict and the domain dict,
        with the updated variable name as key, the bound/domain/init value as value.
    """

    y_bound = {}
    y_domain = {}
    y_init = {}

    for var in var_list:
        y_bound[var.name] = var.bounds
        y_domain[var.name] = var.domain
        y_init[var.name] = var.value

    return y_bound, y_domain, y_init


def _get_x_config(var_list: list, var_name_dict: Dict[str, str]):
    """
    Get bounds, domains, and initial values for second-stage variables.

    Args:
        var_list: The flattened variable list.
        var_name_dict: The renamed variable dict.

    Returns:
        tuple(dict, dict): The variable bound dict and the domain dict.
    """

    x_bound = {}
    x_domain = {}
    x_init = {}

    for var in var_list:

        # extract scenario
        indices = var.name.split('[')[1].split(']')[0]
        if ',' in indices:
            s_idx = indices.split(',')[-1]
        else:
            s_idx = indices

        try:
            s_idx = int(s_idx)
        except ValueError:
            pass

        x_bound[s_idx, var_name_dict[var.name]] = var.bounds
        x_domain[s_idx, var_name_dict[var.name]] = var.domain
        x_init[s_idx, var_name_dict[var.name]] = var.value

    return x_bound, x_domain, x_init


def _get_con2_map(con2_list: list, con2_name_map: Dict[str, str]):
    """
    Get the map for second-stage constraints.

    Args:
        con2_list: The flattened constraint list.
        con2_name_map: The renamed constraint dict.

    Returns:
        dict: The constraint dict.
    """

    con2_dict = {}

    for con in con2_list:

        # extract scenario
        indices = con.name.split('[')[1].split(']')[0]
        if ',' in indices:
            s_idx = indices.split(',')[-1]
        else:
            s_idx = indices
        try:
            s_idx = int(s_idx)
        except ValueError:
            pass

        # create the first key
        if s_idx not in con2_dict:
            con2_dict[s_idx] = {}

        con2_dict[s_idx][con2_name_map[con.name]] = con

    return con2_dict


def _get_var1_map(old_var1: list, new_model: ConcreteModel):
    """
    Map the original (scalar) first-stage variable name to the new model
    variable.

    Args:
        m: The new Pyomo model.
    """
    var1_map = {}

    for var in old_var1:
        for _idx in var.index_set():

            if _idx is not None:

                try:  # single indexed by scenario
                    new_var_name = var.name + '[' + ','.join(_idx) + ']'

                except TypeError:  # multi indexed
                    new_var_name = var.name + '[' + str(_idx) + ']'

            else:
                new_var_name = var.name

            var1_map[id(var[_idx])] = new_model.y[new_var_name]

    return var1_map


def _get_var2_map(old_var2: list, new_model: ConcreteModel):
    """
    Map the original (scalar) second-stage variable name to the new model
    variable.

    Args:
        m: The new Pyomo model.
    """

    var2_map = {}

    for var in old_var2:
        for _idx in var.index_set():

            try:  # single indexed by scenario
                s_idx = _idx[-1]
                other_idx = _idx[:-1]
                new_var_name = var.name + \
                    '[' + ','.join([str(idx)for idx in other_idx]) + ']'

            except TypeError:  # multi indexed
                s_idx = _idx
                new_var_name = var.name

            var2_map[id(var[_idx])] = new_model.x[s_idx,
                                                  new_var_name]

    return var2_map
