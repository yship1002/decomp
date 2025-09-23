from src.analyses.value_function import ValueFunction
from src.analyses.convergence_analysis import HausdorffAnalyzer
from pyomo.opt import SolverFactory
import copy
import matplotlib.pyplot as plt
import numpy as np
def plotting(m,steps,y_dimension,y_sol,y_bound,y_obj):
    # Fix y to the optimal solution to compute the scenario optimal obj
    m.update_y_bound({i:[y_sol[i],y_sol[i]] for i in y_sol})
    scenario_optimal_obj={s: 0 for s in m.scenarios}
    for s in m.scenarios:
        solver=SolverFactory('baron')
        results=solver.solve(m.aux_models['lbd'][s])
        scenario_optimal_obj[s] = results.problem[0]['Upper bound']
    
    # Reset y bounds
    m.update_y_bound(y_bound)
    
    # User supplied y_dimension should be a dictionary of the form
    # {'y_var_name1': {}, 'y_var_name2': {}, ...}
    # where the keys are the names of the y variables to plot and the values are empty dictionaries
    # that will be filled with scenario value functions and total value function
    for y in y_dimension:
        v_f= ValueFunction(m,solver="baron")
        v_f.calc_1D(idx=y, y_val_fix=copy.deepcopy(y_sol),
                        interval= y_bound[y], step = steps)
        for s in m.scenarios:
            y_dimension[y][s] = v_f.value_func[s]
        y_dimension[y]['total'] = v_f.total_value_func
    

    fig, axs = plt.subplots(
        nrows=len(y_dimension), 
        ncols=len(m.scenarios) + 1, 
        figsize=(5 * (len(m.scenarios) + 1), 5 * len(y_dimension))
    )
    for row_idx, (y_key, y_data) in enumerate(y_dimension.items()):
        # Plotting scenario columns
        # ensure the given y_val_fix[idx] is in the discretization
        interval_dis = np.linspace(*y_bound[y_key], num=steps) # type: ignore
        replaceval=y_sol[y_key]
        replaceidx=np.searchsorted(interval_dis,replaceval)
        interval_dis[replaceidx] = replaceval
        for col_idx, s in enumerate(m.scenarios):
            ax = axs[row_idx, col_idx] if len(y_dimension) > 1 else axs[col_idx]
            y_data[s] = np.where(np.isinf(y_data[s]), np.nan, y_data[s])

            ax.plot(
                interval_dis, 
                y_data[s]
            )
            # put red dot at global solution
            findidx=np.where(interval_dis==y_sol[y_key])[0][0]
            ax.scatter(y_sol[y_key], y_data[s][findidx], marker='o', color='red', s=100, label='Global Solution')
            
            ax.set_xlim(y_bound[y_key])
            ax.set_title(f"{y_key} - Scenario {s}", fontsize=20)
            ax.set_xlabel(y_key, fontsize=20)
            ax.set_ylabel("Value", fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.legend()

        # Plot total in the last column
        ax = axs[row_idx, -1] if len(y_dimension) > 1 else axs[-1]

        ax.plot(
            interval_dis, 
            y_data['total'],  color='black'
        )
        # put red dot at global solution
        findidx=np.where(interval_dis==y_sol[y_key])[0][0]
        ax.scatter(y_sol[y_key], y_data["total"][findidx], marker='o', color='red', s=100, label='Global Solution')

        ax.set_xlim(y_bound[y_key])
        ax.set_title(f"{y_key} - Total", fontsize=20)
        ax.set_xlabel(y_key, fontsize=20)
        ax.set_ylabel("Value", fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.legend()
    plt.tight_layout()
    return fig