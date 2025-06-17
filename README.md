# DecompConv: Convergence Rate Analysis Package on Decomposition Methods for Stochastic Optimization

This repository contains the code for analyzing existing decomposition methods
that globally solve nonconvex stochastic optimization problems.
Voila:

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yship1002/decomp/HEAD?urlpath=voila%2Frender%2Flauncher.ipynb)

Regular:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yship1002/decomp/HEAD)

## Main Features
- Partially reproduced the bounding schemes in two state-of-the-art decomposition algorithms: [Cao & Zavala](https://link.springer.com/article/10.1007/s10898-019-00769-y) and [Li & Grossmann](https://link.springer.com/article/10.1007/s10898-019-00816-8)
- Implemented multiple analysis tools, including Hausdorff metric calculation, value function visualization, etc.
- Equipped with the problem library NSPLIB
- Extended the subgradient method in the LG method with various rules

## `main` Branch Layout
```
.
├── NSPLIB                              # library of nonconvex stochastic optimization problems
├── src
│   ├── analyses
│   │   ├── __init__.py
│   │   ├── convergence_analysis.py     # analyze Hausdorff convergence
│   │   └── value_function.py           # plot value function in 1/2D
│   ├── models
│   │   ├── __init__.py
│   │   ├── bb_node.py                  # classes for branch-and-bound nodes
│   │   ├── cz_model.py                 # Cao & Zavala method class
│   │   ├── decomp_model.py             # base decomposition method class
│   │   ├── deflection_rules.py         # deflection rules for subgradient method
│   │   ├── lagrangean_model.py         # (idealized) Lagrangean method class
│   │   ├── stepsize_rules.py           # stepsize rules for subgradient method
│   │   └── subgradient_method.py       # main class for subgradient method
│   ├── utility
│   ├── __init__.py
│   └── config.ini                      # configure python loggers
├── tests
├── .gitmodules
├── README.md
└── requirements.txt
```

## Usage (to be updated)

See `results/CZ/pooling.ipynb` as an example.
Submodule 'pyomo' is a modified version where results.solver.root_node_time attr is added for baron

## Solver Usage

This package utilizes several solvers that need to be set up separately.

- LP/MILP/MIQP problems:
    - [Gurobi](https://www.gurobi.com), free academic license available
- general MINLP problems:
    - [BARON](https://minlp.com/baron-solver), free academic license available for USG affiliates
    - [SCIP](https://www.scipopt.org), open source
