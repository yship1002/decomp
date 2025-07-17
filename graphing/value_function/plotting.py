from src.analyses.value_function import ValueFunction
from src.analyses.convergence_analysis import HausdorffAnalyzer
from pyomo.opt import SolverFactory
import copy
import matplotlib.pyplot as plt
import numpy as np
def plotting(m,steps,y_dimension,y_sol,y_bound,y_obj):
    m.update_y_bound({i:[y_sol[i],y_sol[i]] for i in y_sol})
    scenario_optimal_obj={s: 0 for s in m.scenarios}
    for s in m.scenarios:
        solver=SolverFactory('baron')
        results=solver.solve(m.aux_models['lbd'][s])
        scenario_optimal_obj[s] = results.problem[0]['Upper bound']
    m.update_y_bound(y_bound)
    
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
        for col_idx, s in enumerate(m.scenarios):
            ax = axs[row_idx, col_idx] if len(y_dimension) > 1 else axs[col_idx]
            y_data[s] = np.where(np.isinf(y_data[s]), np.nan, y_data[s])
            ax.plot(
                np.linspace(*y_bound[y_key], steps), 
                y_data[s]
            )
            ax.scatter(y_sol[y_key],scenario_optimal_obj[s],marker='o', color='red', s=100, label='Global Solution')
            ax.set_xlim(y_bound[y_key])
            ax.set_title(f"{y_key} - Scenario {s}", fontsize=16)
            ax.set_xlabel(y_key, fontsize=14)
            ax.set_ylabel("Value", fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend()
        # Plot total in the last column
        ax = axs[row_idx, -1] if len(y_dimension) > 1 else axs[-1]
        ax.plot(
            np.linspace(*y_bound[y_key], steps), 
            y_data['total'],  color='black'
        )
        ax.scatter(y_sol[y_key], y_obj, marker='o', color='red', s=100, label='Global Solution')
        ax.set_xlim(y_bound[y_key])
        ax.set_title(f"{y_key} - Total", fontsize=16)
        ax.set_xlabel(y_key, fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.legend()

    plt.tight_layout()
    plt.show()
    return y_dimension,scenario_optimal_obj