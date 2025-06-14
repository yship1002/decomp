# Nonconvex Stochastic Programming LIBrary (STPLIB)

<!-- TODO add package introduction -->

<p align="center"><img src="svgs/5f1267869f6ca307e518d0d55300d55c.svg?invert_in_darkmode" align=middle width=236.30676629999996pt height=136.97139719999998pt/></p>
<!-- $$ \begin{aligned} \min_{\mathbf{y}, \mathbf{x}1, \dots, \mathbf{x}{|S|}} & \sum_{s\in S} f_s(\mathbf{y}, \mathbf{x}_s) \ \mathrm{s.t.} \ \ & \mathbf{g}_0(\mathbf{y}) \leq \mathbf{0}\ &\mathbf{g}_s(\mathbf{y},\mathbf{x}_s) \leq \mathbf{0}, \quad \forall s \in S \ & \mathbf{y} \in Y \ & \mathbf{x}_s \in X_s, \quad \forall s \in S \end{aligned} $$ -->

## Usage

<!-- TODO add build from pyomo -->
### Create A Stochastic Model from Scratch

1. set indices for scenarios (<img src="svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode" align=middle width=11.027402099999989pt height=22.465723500000017pt/>), first-stage variables (<img src="svgs/1da18d2de6d16a18e780cd6c435a2936.svg?invert_in_darkmode" align=middle width=10.239687149999991pt height=14.611878600000017pt/>), and second-stage variables (<img src="svgs/e9498cd8b56803cac1bce795f5cf6e89.svg?invert_in_darkmode" align=middle width=16.18148729999999pt height=14.611878600000017pt/>)
```python
# USAGE: create list for index sets for scenarios, first-stage variables (y),
# and second-stage variables (x). Then call `.set_sets()`.

from sto_model import StochasticModel

# create model
m = StochasticModel()

scenarios = [3, 4]
y_set = [0]
x_set = [1, 2]
m.set_sets(scenarios, y_set, x_set)
```

2. set objective functions (<img src="svgs/deb18c89b908abf80bef809cbdcbae2d.svg?invert_in_darkmode" align=middle width=14.252356799999989pt height=22.831056599999986pt/>)
```python
# USAGE: create a objective function dict, with scenario as key, the objective
# function as value. Then call `.set_obj()`. The objective function format
# should be similar to the one for setting objective in Pyomo (i.e., a native
# python function returning expression), except that it should have scenario as
# the second argument.

# NOTE: x is indexed by two sets: the scenario index (as the first one), and the x index (the member within x_set).
def f3(m, s):
    return m.y[0] + m.x[s, 1] + m.x[s, 2]

def f4(m, s):
    return m.y[0] - m.x[s, 2] + 2 * m.x[s, 1]

# function dict, scenario as key
f_s = {
    3: f3, 
    4: f4
}

m.set_obj(f_s)
```
3. set constraints w.r.t. only first-stage variables (<img src="svgs/df572e8d3298a34611ae35a79acc0f63.svg?invert_in_darkmode" align=middle width=14.771756999999988pt height=14.15524440000002pt/>)
```python
# USAGE: define constraints in this category as native python functions (similar
# to defining constraints in Pyomo). Then wrap these functions into a list and
# call `.set_con_stage_1()`.

def g_0_1(m):
    return m.y[0] <= 0.5

def g_0_2(m):
    return m.y[0] >= -0.5

g_0 = [g_0_1, g_0_2]
m.set_con_stage_1(g_0)
```
4. set constraints w.r.t. second-stage variables (<img src="svgs/09f01c659cc6531538060d06c28ece12.svg?invert_in_darkmode" align=middle width=14.42358059999999pt height=14.15524440000002pt/>)
```python
# USAGE: define constraints in this category as native python functions. Then
# form a dict with scenario index as key, a list of constraints that are active
# in the scenario as value. Finally, call `.set_con_stage_2()`.

# NOTE: Functions for constraints in this category should have scenario index as
# the second argument.
def g_3_1(m, s):
    return m.y[0] + m.x[s, 2] <= 1

def g_3_2(m, s):
    return m.x[s, 1] + m.x[s, 2] >= -10

def g_4_1(m, s):
    return m.x[s, 1] + m.x[s, 2] <= 10

# constraint dict, scenario as key
g_s = {
    3: [g_3_1, g_3_2],
    4: [g_4_1]
}

m.set_con_stage_2(g_s)
```
5. set bounds for variables (<img src="svgs/89a9b40613b97caf4234a65daff70c51.svg?invert_in_darkmode" align=middle width=37.58565029999999pt height=22.465723500000017pt/>)
<!-- set indices for scenarios ($S$), first-stage variables ($\mathbf{y}$), and second-stage variables ($\mathbf{x}_s$)
set objective functions ($f_s$)
set constraints w.r.t. only first-stage variables ($\mathrm{g}_0$)
set constraints w.r.t. second-stage variables ($\mathrm{g}_s$)
set bounds for variables ($Y, X_s$) -->
```python
# USAGE: from dicts for setting variable bounds. For y, use y index as key,
# (lower bound, upper bound) as value. for x, use scenario as the first key, 
# x index as the second key, (lower bound, upper bound) as value. 

# set y bound
Y = {
    0: [-1, 1]
}

m.set_y_bound(Y)

# set x bound
X = {
    3: {
        1: [-1, 0],
        2: [0, 1]
    },
    4: {
        1: [2, 3], 
        2: [4, 5]
    }
}
m.set_x_bound(X)
```
6. build and access model
```python
# build pyomo model
m.build()

# access the pyomo model
m.pyomo_model
```

## `main` Branch Layout
```
.
├── src
│   ├── instances
│   │   ├── illustrative_examples       # a set of small-scale problems
│   │   ├── PID                         # tuning controller
│   │   └── pooling_contract_selection  # pooling problem + contract selection
│   └── main.py                         # for main class `StochasticModel`
├── svgs                                # store LaTeX-derived figures for README
├── tests
└── README.md
```
