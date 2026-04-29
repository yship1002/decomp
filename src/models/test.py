from pyomo_to_mcpp import constraint_to_mcpp
from pyomo.environ import ConcreteModel, Var, Constraint
m = ConcreteModel()
m.x = Var(bounds=(-2, 2))
m.y = Var(bounds=(0, 4))
m.c = Constraint(expr=m.x**2 + 2*m.x*m.y <= 5)

print(constraint_to_mcpp(m.c))