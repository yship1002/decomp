import matplotlib.pyplot as plt
import numpy as np


def plot_converge_order(eps_list, distances, metric='Hausdorff', x_tick_n=6):
    """Plot the convergence order.

    Args:
        eps_list (list): The list of interval diameters.
        distances (list): The list of Hausdorff distances.
        metric (str, optional): The convergence metric. Defaults to 'Hausdorff'.
        x_tick_n (int, optional): The number of x ticks. Defaults to 6.

    Returns:
        (fig, ax): The convergence plot.
    """

    # use predefined style
    plt.style.use(['./src/utility/' + i + '.mplstyle' for i in ['font-sans', 'size-4-4', 'fontsize-12']])

    fig, ax = plt.subplots()

    # log-log plot for normal situation
    if sum(distances) > 0:
        ax.loglog(eps_list, distances, 'b-')
    # semi-log-x for all-zero case
    else:
        ax.semilogx(eps_list, distances, 'b-')

    # plot grid
    ax.grid(True, which='major', axis='both')
    # remove minor ticks
    ax.minorticks_off()

    # generate x ticks
    eps_min, eps_max = min(eps_list), max(eps_list)
    # get power of 10's for max and min ticks
    tick_min = find_largest_power_10_smaller_than_equal(eps_min)
    tick_max = find_smallest_power_10_larger_than_equal(eps_max)
    tick_min_log = round(np.log10(tick_min))
    tick_max_log = round(np.log10(tick_max))
    power_diff = tick_max_log - tick_min_log + 1
    ticks = np.logspace(tick_min_log, tick_max_log, min(x_tick_n, power_diff))
    ax.set_xticks(ticks)

    # set labels
    ax.set_xlabel(r"$\mathrm{diam}(Y)$")
    ax.set_ylabel(f"{metric} metric")

    return fig, ax


def find_largest_power_10_smaller_than_equal(x):

    res = 1

    if x >= 1:
        while res * 10 <= x:
            res *= 10
        return res
    else:
        multiplier = 1
        while res > x * multiplier:
            multiplier *= 10
        return res / multiplier


def find_smallest_power_10_larger_than_equal(x):

    res = 1

    if x >= 1:
        while res < x:
            res *= 10
        return res
    else:
        while res * 0.1 >= x:
            res *= 0.1
        return res
