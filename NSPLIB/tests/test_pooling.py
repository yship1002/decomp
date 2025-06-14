from unittest import TestCase
from src.main import StochasticModel, replace_expression
from src.instances.pooling_contract_selection.pooling import const_model
from pyomo.environ import *

class TestPooling(TestCase):

    def test_reconstructed_sol(self):
        """Test if the reformulated model gives the same objective value."""

        m = const_model()
        mm = m.pyomo_model

        opt = SolverFactory('gurobi')
        opt.options['NonConvex'] = 2
        opt.options['MIPGap'] = 1e-5

        res = opt.solve(mm, tee=True)

        self.assertAlmostEqual(value(mm.obj), -1338.234071658, delta=0.01)
