from unittest import TestCase
# from src.models.decomp_model import DecompModel
from src.models.cz_model import CaoZavalaModel
from pyomo.environ import *

class TestDecompModel(TestCase):

    def test_set_gurobi(self):

        m = CaoZavalaModel(solver='gurobi')

        self.assertEqual(m.solver.name, 'gurobi')
        self.assertEqual(m.solver.spec['options']['NonConvex'], 2)
    
    def test_set_var(self):

        m = ConcreteModel()
        m.x = Var()
        pass