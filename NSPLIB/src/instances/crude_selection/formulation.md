# Stochastic pooling problem with contract selection

## source
- https://link.springer.com/article/10.1007/s10898-019-00816-8
- https://github.com/bbrunaud/PlasmoAlgorithms.jl/blob/clrootnode/examples/refinery_model/continous/scaled_refinery_continous.gms
- Can Li, Ignacio Grossmann
- based on the example in Yang et al., AIChE J., 2016

## Formulation

<!-- See Appendix 1 of the paper.
This instance corresponds to the 3-scenario one. -->

## Optimal Solution
The following solution is for `NS = 5`.
Gurobi is set to solve to $0.001\%$ optimality.
- pickCrude[i] = 1 for i in {2, 3, 4, 8, 10}
- crudeQuantity[2] = 152.08699870406969 
- crudeQuantity[3] = 201.29570747217807 
- crudeQuantity[4] = 55.933433970725986 
- crudeQuantity[8] = 164.54868508853025 
- crudeQuantity[10] = 15.55927487524582 
- optimal objective value: -18351.05515105 (lbd) / -18350.09018181 (ubd)
- total CPU time: 4.36 s