# Stochastic pooling problem with contract selection

## source
- https://link.springer.com/article/10.1007/s10898-019-00816-8
- https://github.com/bbrunaud/PlasmoAlgorithms.jl/blob/clrootnode/examples/PoolingContract/fullspace3.gms
- Can Li, Ignacio Grossmann

## Formulation

See Appendix 1 of the paper.
This instance corresponds to the 3-scenario one.

## Optimal Solution
The following solution is for `NS = 3`.
Gurobi is set to solve to $0.001\%$ optimality.
- lambda[1] = lambda[2] = lambda[5] = 1
- A[1] = 300, A[2] = 201.9218, A[5] = 245.1782
- theta[1] = theta[4] = 1
- S[1] = 247.1298, S[4] = 499.9689
- optimal objective value: -1338.247128338 (lbd) / -1338.234071658 (ubd)
- total CPU time: 4.36 s