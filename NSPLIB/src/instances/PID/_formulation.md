# Optimal Controller Tuning

## source
- https://link.springer.com/article/10.1007/s10898-019-00769-y
- https://github.com/zavalab/JuliaBox/blob/master/SNGO/examples/PID/pidnonlinear.jl
- Yankai Cao, Victor Zavala

## Problem Statement

The objective of this problem is to tune the controller that needs to deal with
different scenarios with different disturbances, set point changes, and structural
uncertainties.
The following formulation models the problem as a two-stage SP problem where
the first-stage variables are the controller gains, and the second-stage ones
are the state time trajectories, where the time is discretized.

When generating instance, the model function takes two inputs: `NS` for the
number of scenarios, and `N` for the number of time points.
In each scenario, the disturbance, set point change, and structural
uncertainties follow certain uniform distributions.
In the current function, the randomness is fixed to `np.random.seed(0)`.

## Formulation

$$
\begin{aligned}
    \min \ & \sum_{s \in S} \bar{c}_s \\
    \mathrm{s.t.} \ & \frac{1}{\tau^x_{s}} \cdot (x_{s, t + 1} - x_{s, t}) / h +
    (x_{s, t + 1})^2 = \tau^u_s \cdot u_{s, t + 1} + \tau^d_s \cdot d_{s, t},
    && \quad \forall s \in S, t \in T \setminus \{|T| - 1\} \\
    & u_{s, t + 1} = K_p \cdot (\bar{x}_s - x_{s, t}) + K_i \cdot I_{s, t + 1} + K_d \cdot (x_{s, t + 1} - x_{s, t}) / h,
    && \quad \forall s \in S, t \in T \setminus \{|T| - 1\} \\
    & (I_{s, t + 1} - I_{s, t}) / h = \bar{x}_s - x_{s, t + 1},
    && \quad \forall s \in S, t \in T \setminus \{|T| - 1\} \\
    & x_{s, 0} = x_0,
    && \quad \forall s \in S \\
    & I_{s, 0} = 0,
    && \quad \forall s \in S \\
    & c_{s, 0} = 10 \cdot (x_{s, 0} - \bar{x}_s)^2,
    && \quad \forall s \in S \\
    & c_{s, t} = 10 \cdot (x_{s, t} - \bar{x}_s)^ 2 + 0.01 \cdot u_{s, t}^2,
    && \quad \forall s \in S, t \in T \setminus \{0\} \\
    & \bar{c}_{s} = \frac{100}{|T| \cdot |S|} \cdot \sum_{t \in T} c_{s, t},
    && \quad \forall s \in S \\
    & K_p \in [-10, 10], K_i \in [-100, 100], K_d \in [-100, 1000] \\
    & x_{s, t} \in [-2.5, 2.5],
    && \quad \forall s \in S, t \in T \\
    & u_{s, t} \in [-5, 5],
    && \quad \forall s \in S, t \in T\setminus\{0\} \\
\end{aligned}
$$

## Nomenclature

### Sets

- $S$: set of scenarios, index starting from 0; determined by the input `NS`
- $T$: set of discretized time points, index starting from 0; determined by the input `N`

### Parameters
For each scenario, these parameters follows uniform distributions independently.
- $\bar{x}_s$: set point change; $[-2.3, 2.3]$
- $\tau^x_s$, $\tau^u_s$, $\tau^d_s$: model structural uncertainty;
$[0.2, 0.8], [1, 5], [0.2, 0.8]$
- $d_s$: disturbance; $[-2.5, 2.5]$
- $h$: length of each time period, $15/\texttt{N}$
- $x_0$: initial condition, $0$

### Variables
#### First-Stage
- $K_{p}$: gain
- $K_{i}$: integral gain
- $K_{d}$: derivative gain
#### Second-Stage
- $x_{s, t}$: state time trajectory
- $u_{s, t}$: error
- $I_{s, t}$: integrated/accumulated error
- $c_{s, t}$: cost (square of error) of each time point
- $c_{s, t}$: overall cost


## Optimal Solution
The following solution is for `N = 20, NS = 10, np.random.seed(0)`.
Baron is set to solve to $0.01\%$ optimality.
- Kp = 10
- Ki = 100
- Kd = -6.873680693108097
- optimal objective value: 104.991738158 (lbd) / 105.00254416 (ubd)
- total CPU time: 29.32 s
- When the relative optimality was set to `1e-6`, Baron failed to improve the
gap for 1 hr.