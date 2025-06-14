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

<p align="center"><img src="svgs/338623af8554be434f1a818c79a5ea7f.svg?invert_in_darkmode" align=middle width=678.31513035pt height=361.94758379999996pt/></p>

## Nomenclature

### Sets

- <img src="svgs/e257acd1ccbe7fcb654708f1a866bfe9.svg?invert_in_darkmode" align=middle width=11.027402099999989pt height=22.465723500000017pt/>: set of scenarios, index starting from 0; determined by the input `NS`
- <img src="svgs/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode" align=middle width=11.889314249999991pt height=22.465723500000017pt/>: set of discretized time points, index starting from 0; determined by the input `N`

### Parameters
For each scenario, these parameters follows uniform distributions independently.
- <img src="svgs/9f1c9ba7bee8c3bf83c76b4d50d8bce3.svg?invert_in_darkmode" align=middle width=15.599359049999991pt height=18.666631500000015pt/>: set point change; <img src="svgs/4b2dc2886fca03bdb517b6db5707d4df.svg?invert_in_darkmode" align=middle width=71.23305089999998pt height=24.65753399999998pt/>
- <img src="svgs/9c8d111a8b3e0c698031767842918525.svg?invert_in_darkmode" align=middle width=16.50120614999999pt height=21.839370299999988pt/>, <img src="svgs/a3ba801b94eacabb7656f824d84d7e7c.svg?invert_in_darkmode" align=middle width=16.81898459999999pt height=21.839370299999988pt/>, <img src="svgs/4b7684a5421606d7967d944d845faeab.svg?invert_in_darkmode" align=middle width=15.889910849999989pt height=27.91243950000002pt/>: model structural uncertainty;
<img src="svgs/9d90d98f26e8c2fc9f38bf9ee22ec619.svg?invert_in_darkmode" align=middle width=164.38374975pt height=24.65753399999998pt/>
- <img src="svgs/3f2c93f4db0df26af089195b2ac29c1c.svg?invert_in_darkmode" align=middle width=14.760334049999992pt height=22.831056599999986pt/>: disturbance; <img src="svgs/feb3da384aaf71db972449fca0477c72.svg?invert_in_darkmode" align=middle width=71.23305089999998pt height=24.65753399999998pt/>
- <img src="svgs/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode" align=middle width=9.47111549999999pt height=22.831056599999986pt/>: length of each time period, <img src="svgs/8e9b814792d0b7d93f2de47f4034c737.svg?invert_in_darkmode" align=middle width=33.28769564999999pt height=24.65753399999998pt/>
- <img src="svgs/e714a3139958da04b41e3e607a544455.svg?invert_in_darkmode" align=middle width=15.94753544999999pt height=14.15524440000002pt/>: initial condition, <img src="svgs/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/>

### Variables
#### First-Stage
- <img src="svgs/4b68d6a3baa78c597954163a94a91314.svg?invert_in_darkmode" align=middle width=20.73769004999999pt height=22.465723500000017pt/>: gain
- <img src="svgs/fe4f49408e90de8b6972a9d7998a620d.svg?invert_in_darkmode" align=middle width=18.61211054999999pt height=22.465723500000017pt/>: integral gain
- <img src="svgs/ea70c858d28cf7b519061a05d1cefa4b.svg?invert_in_darkmode" align=middle width=20.804288999999986pt height=22.465723500000017pt/>: derivative gain
#### Second-Stage
- <img src="svgs/933f5e854b0654f04631a1fc3f1cf4eb.svg?invert_in_darkmode" align=middle width=24.46928384999999pt height=14.15524440000002pt/>: state time trajectory
- <img src="svgs/0042cace08774ed72741e4a26b666296.svg?invert_in_darkmode" align=middle width=24.484569449999988pt height=14.15524440000002pt/>: error
- <img src="svgs/dc8b987e7a18c15b7636723e3b918115.svg?invert_in_darkmode" align=middle width=22.30034564999999pt height=22.465723500000017pt/>: integrated/accumulated error
- <img src="svgs/fc858a72acdd89cc26677fa1602c5d20.svg?invert_in_darkmode" align=middle width=22.18809944999999pt height=14.15524440000002pt/>: cost (square of error) of each time point
- <img src="svgs/fc858a72acdd89cc26677fa1602c5d20.svg?invert_in_darkmode" align=middle width=22.18809944999999pt height=14.15524440000002pt/>: overall cost


## Optimal Solution
The following solution is for `N = 20, NS = 10, np.random.seed(0)`.
Baron is set to solve to <img src="svgs/32a0f4e49c00cf15313672f10bf55aa9.svg?invert_in_darkmode" align=middle width=42.922524149999994pt height=24.65753399999998pt/> optimality.
- Kp = 10
- Ki = 100
- Kd = -6.873680693108097
- optimal objective value: 104.991738158 (lbd) / 105.00254416 (ubd)
- total CPU time: 29.32 s
- When the relative optimality was set to `1e-6`, Baron failed to improve the
gap for 1 hr.