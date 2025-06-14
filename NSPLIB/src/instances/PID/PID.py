"""
Features
    -

The notations in this script are closer to the ones in the paper.

correspondence of variables in Zavala GitHub (left) & paper (right)
    first-stage variables
        Kc -------- Kp
        tauI ------ Ki
        tauD ------ Kd
    second-stage variables
        x --------- x_s(t)
        u --------- u_s(t)
        int ------- (None)
        cost ------ (None)
        costS ----- (None)
        (None) ---- e_s(t)
    parameters
        xsp ------- \bar{x}_s
        d --------- d_s
        tau ------- tau_{x, s}
        K --------- tau_{u, s}
        Kd -------- tau_{d, s}
        h --------- (None)      # time horizon

Source
    https://link.springer.com/article/10.1007/s10898-019-00769-y
    https://github.com/zavalab/JuliaBox/blob/master/SNGO/examples/PID/pidnonlinear.jl
    Yankai Cao, Victor Zavala

"""

import numpy as np
from ...main import StoModelBuilder


np.random.seed(0)


def const_model(N, NS):
    """

    Args:
        N (int): The number of time steps.
        NS (int): The number of scenarios.
    """

    # time step set
    T = range(N)
    # time step set excluding the last step
    Tm = range(N - 1)
    # time step set excluding the first step
    mT = range(1, N)

    # total time?
    Tf = 15
    # horizon
    h = Tf / N

    # initial step
    x0 = 0

    # pre-defined value for the first 10 scenarios
    _tauU = np.array([
        5.0,
        1.0,
        2.0,
        1.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        1.5
    ])
    _tauD = np.array([
        0.5,
        0.4,
        0.3,
        0.5,
        0.8,
        0.5,
        0.4,
        0.3,
        0.7,
        0.25
    ])
    _tauX = np.array([
        0.5,
        0.5,
        0.3,
        0.4,
        0.7,
        0.5,
        0.5,
        0.3,
        0.4,
        0.6
    ])
    _xsp = np.array([
        -1.0,
        1.0,
        -2.0,
        1.5,
        2.0,
        -1.5,
        1.5,
        0,
        1.5,
        2.0
    ])
    _d = np.zeros((NS, N))
    _d[0, :] = -1
    _d[1, :] = +1
    _d[2, :] = 2
    _d[3, :] = -2
    _d[4, :] = -1
    _d[5, :] = -0.5
    _d[6, :] = +0.5
    _d[7, :] = 2.5
    _d[8, :] = -1.5
    _d[9, :] = -1

    tauU = np.zeros(NS)
    tauD = np.zeros(NS)
    tauX = np.zeros(NS)
    xsp = np.zeros(NS)
    d = np.zeros((NS, N))
    # apply the predefined values
    # tauU[:min(NS, 10)] = _tauU[:min(NS, 10)]
    # tauD[:min(NS, 10)] = _tauD[:min(NS, 10)]
    # tauX[:min(NS, 10)] = _tauX[:min(NS, 10)]
    # xsp[:min(NS, 10)] = _xsp[:min(NS, 10)]
    # d[:min(NS, 10), :] = _d[:min(NS, 10), :]
    # for j in range(10, NS):
    # generate parameters for each scenario
    for j in range(NS):
        tauU[j] = np.random.uniform(1, 5)
        tauD[j] = np.random.uniform(0.2, 0.8)
        tauX[j] = np.random.uniform(0.2, 0.8)
        xsp[j] = np.random.uniform(-2.3, 2.3)
        d[j, :] = np.random.uniform(-2.5, 2.5)

    # first-stage variables: gain Kp, integral gain Ki, derivative gain Kd
    y_set = ['Kp', 'Ki', 'Kd']

    # second-stage variables: integral, error, x, cost, costS
    # explicitly claim separate variables for each time point
    x_set = ['x' + str(i) for i in T]
    x_set += ['u' + str(i) for i in mT]
    x_set += ['int' + str(i) for i in T]
    x_set += ['cost' + str(i) for i in T]
    x_set += ['costS']

    scenarios = list(range(NS))

    Y = {
        'Kp': [-10, 10],
        'Ki': [-100, 100],
        'Kd': [-100, 1000],
    }

    X = {}
    for s in scenarios:
        X[s, 'costS'] = [None, None]
        for i in T:
            X[s, 'x' + str(i)] = [-2.5, 2.5]
            X[s, 'int' + str(i)] = [None, None]
            X[s, 'cost' + str(i)] = [None, None]
        for i in mT:
            X[s, 'u' + str(i)] = [-5, 5]

    # Eq. (5.1b) variant - discretized
    def con1_curry(t):
        def con1(m, s):
            return (1 / tauX[s]) * (m.x[s, 'x' + str(t + 1)] - m.x[s, 'x' + str(t)]) / h + \
                m.x[s, 'x' + str(t + 1)] * m.x[s, 'x' + str(t + 1)] == \
                tauU[s] * m.x[s, 'u' + str(t + 1)] + \
                tauD[s] * d[s, t]
        return con1

    # Eq. (5.1d)
    def con2_curry(t):
        def con2(m, s):
            return m.x[s, 'u' + str(t + 1)] == \
                m.y['Kp'] * (xsp[s] - m.x[s, 'x' + str(t)]) + \
                m.y['Ki'] * m.x[s, 'int' + str(t + 1)] + \
                m.y['Kd'] * (m.x[s, 'x' + str(t + 1)] -
                             m.x[s, 'x' + str(t)]) / h
        return con2

    # Eq. (5.1c) variant - directly calculate integral
    def con3_curry(t):
        def con3(m, s):
            return \
                (m.x[s, 'int' + str(t + 1)] - m.x[s, 'int' + str(t)]) / h == \
                xsp[s] - m.x[s, 'x' + str(t + 1)]
        return con3

    # initial condition
    def con4(m, s):
        return m.x[s, 'x0'] == x0

    # initial condition
    def con5(m, s):
        return m.x[s, 'int0'] == 0

    # initial condition
    def con6(m, s):
        return m.x[s, 'cost0'] == 10 * (m.x[s, 'x0'] - xsp[s]) ** 2

    # calculation of "cost" term
    def con7_curry(t):
        def con7(m, s):
            return \
                m.x[s, 'cost' + str(t)] == \
                10 * (m.x[s, 'x' + str(t)] - xsp[s]) ** 2 + \
                0.01 * m.x[s, 'u' + str(t)] ** 2
        return con7

    # calculation of total "cost"
    def con8(m, s):
        return m.x[s, 'costS'] == 100 / len(T) / NS * sum(m.x[s, 'cost' + str(i)] for i in T)

    g_s = {
        s: [con4, con5, con6, con8]
        + [con1_curry(t) for t in Tm]
        + [con2_curry(t) for t in Tm]
        + [con3_curry(t) for t in Tm]
        + [con7_curry(t) for t in mT]
        for s in scenarios
    }

    def _obj(m, s):
        return m.x[s, 'costS']

    objs = {s: _obj for s in scenarios}

    builder = StoModelBuilder('direct', name='PID', m_type='NLP', hint=False)

    _content = {
        'y_set': y_set,
        'x_set': x_set,
        'scenarios': scenarios,
        'con_2': g_s,
        'y_bound': Y,
        'x_bound': X,
        'objs': objs
    }
    sto_m = builder.build(**_content)

    return sto_m
