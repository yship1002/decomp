from unittest import TestCase
from src.models.subgradient_method import SubgradientMethod
from src.models.stepsize_rules import ColorTVRule
from src.models.deflection_rules import STSubgradRule
from src.models.lagrangean_model import LagrangeanModel 


class TestLagrangeanModel(TestCase):

    def test_sm_solve(self):
        from NSPLIB.src.instances.nonlinear_2D import const_model
        sto_m = const_model()
        m = LagrangeanModel.from_sto_m(sto_m)
        m.build()

        stepsize_rule = ColorTVRule()
        deflection_rule = STSubgradRule()
        sm = SubgradientMethod(m.lagrangean_subproblems, m.y_set, stepsize_rule, deflection_rule)

        sm.run()
        sm.lbds
