import matplotlib.pyplot as plt
import numpy as np
def node_plot(result,plot_method='loglog'):
    fathomed_node_n = [i - j for (i, j) in zip(result.bb_n, result.active_bb_n)]
    gap_rec = [1 / g for g in result.abs_gaps]

    plt.style.use(['./src/utility/' + i + '.mplstyle' for i in ['font-sans', 'size-4-4', 'fontsize-12']])

    func_dict = {
        'semilogy': plt.semilogy,
        'loglog': plt.loglog,
        'plot': plt.plot,
    }
    func = func_dict[plot_method]
    lw = 1.75
    func(gap_rec, result.active_bb_n, 'r-', label='active nodes', linewidth=lw)
    func(gap_rec, result.bb_n, 'b-', label='nodes', linewidth=lw)
    func(gap_rec, fathomed_node_n, 'k-', label='fathomed nodes', linewidth=lw)
    plt.xlabel('1 / gap',fontsize=20)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.gca().xaxis.offsetText.set_fontsize(20)
    plt.gcf().set_size_inches(7, 7)
    plt.draw()
    plt.tick_params(axis='x', which='both', labelsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.ylabel('node number',fontsize=20)
    plt.grid(True, which='major', axis='both')
    plt.legend()
    return plt