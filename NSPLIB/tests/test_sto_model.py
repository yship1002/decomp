from unittest import TestCase
from src.main import StochasticModel, replace_expression
from pyomo.environ import *
from unittest.mock import patch
from io import StringIO


class TestReplaceVisitor(TestCase):

    def setUp(self):
        pm = ConcreteModel()

        pm.I = Set(initialize=[1, 2, 3])
        pm.S = Set(initialize=['S1', 'S2'])

        # first-stage variables
        pm.beta = Var(within=Binary)
        pm.sigma = Var(pm.I, within=NonNegativeReals, bounds=(4, 5))
        pm.omega = Var(pm.I, within=NonNegativeReals, 
                       initialize={1: 5, 2: 5, 3: 5})
        # second-stage variables
        pm.delta = Var(pm.S, within=NonNegativeReals, bounds=(7, 9))
        pm.alpha = Var(pm.I, pm.S, within=NonNegativeReals)

        def con1(m):
            return sum(m.omega[i] for i in m.I) >= 1
        pm.con1 = Constraint(rule=con1)

        def con2(m, s):
            return sum(m.alpha[i, s] for i in m.I) >= 1
        pm.con2 = Constraint(pm.S, rule=con2)

        def obj(m):
            # return m.beta
            return sum(m.delta[s] for s in m.S)
        pm.obj = Objective(rule=obj)

        pm.con3 = ConstraintList()
        pm.con3.add(con1(pm))

        self.pm = pm

    def test_replace_terms_in_obj_expr(self):

        m = ConcreteModel()
        m.not_beta = Var()
        var_map = {
            'delta[S1]': m.not_beta,
            'delta[S2]': m.not_beta,
        }

        with patch('sys.stdout', new = StringIO()) as fake_out:
            new_expr = replace_expression(self.pm.obj.expr, var_map)
            print(new_expr)
            self.assertEqual('not_beta + not_beta', fake_out.getvalue()[-20:-1])

    def test_replace_obj(self):
        m = ConcreteModel()
        m.not_beta = Var()
        var_map = {
            'beta': m.not_beta
        }

        new_expr = replace_expression(self.pm.obj.expr, var_map)
        m.obj = Objective(expr=new_expr)
    
    def test_set_multiple_objs(self):
        """Test how to set multiple objective functions to the original model.
        """

        scenarios = ['S1', 'S2']

        def new_obj_sub(m, s):
            return m.delta[s]
        
        for s in scenarios:
            def obj_wrap(m):
                return new_obj_sub(m, s)
            
            setattr(self.pm, 'obj_' + s, Objective(expr=obj_wrap))
    
    def test_replace_multiple_objs(self):
        """The overall replacement test."""

        m = ConcreteModel()
        m.aa = Var()
        m.bb = Var()
        var_map = {
            'delta[S1]': m.aa,
            'delta[S2]': m.bb
        }
        scenarios = ['S1', 'S2']

        def new_obj_sub(m, s):
            return m.delta[s]
        
        for s in scenarios:
            def obj_wrap(m):
                return new_obj_sub(m, s)
            
            setattr(self.pm, 'obj_' + s, Objective(expr=obj_wrap))
            sub_obj_expr = getattr(self.pm, 'obj_' + s)
            new_expr = replace_expression(sub_obj_expr, var_map)
            m.obj = Objective(expr=new_expr)
    
    def test_build_from_pyomo_new_objs(self):

        m = StochasticModel()

        fs_vars = ['beta', 'sigma', 'omega']
        fs_cons = ['con1', 'con3']
        scenarios = ['S1', 'S2']

        def sub_obj(m, s):
            return m.delta[s]
        objs = {s: sub_obj for s in scenarios}

        m.build_from_pyomo(self.pm, fs_vars, fs_cons, scenarios, objs)
        m.pyomo_model.pprint()
    

class TestStochasticModel(TestCase):

    def setUp(self):

        m = StochasticModel()
        self.m = m

        self.scenarios = scenarios = ['S1', 'S2']

        m.alpha = Var()
        m.beta = Var()

        m.a = Var(scenarios)
        m.b = Var(scenarios)

    def test_assign_constraint_con_1(self):

        m = self.m

        def con1(m):
            return m.beta
        
        con_1 = [con1]

        m.set_con_stage_1(con_1)
        self.assertEqual(m.con_stage_1, con_1)
        self.assertEqual(m.con_stage_1_expr, None)

    def test_assign_constraint_con_2(self):

        m = self.m
        m.scenarios = self.scenarios

        def con2(m, s):
            return m.b[s] >= 1
        
        con_2 = {
            'S1': con2,
            'S2': con2,
        }

        m.set_con_stage_2(con_2)
        self.assertEqual(m.con_stage_2, con_2)
        self.assertEqual(m.con_stage_2_expr, None)
    
    def test_assign_constraint_con_2_expr(self):

        m = self.m
        m.scenarios = self.scenarios

        def con21(m):
            return m.b['S1'] >= 1

        def con22(m):
            return m.b['S2'] >= 1
        
        m.__con21 = Constraint(rule=con21)
        m.__con22 = Constraint(rule=con22)
        
        con_2 = {
            'S1': m.__con21.expr,
            'S2': m.__con22.expr,
        }

        m.set_con_stage_2(con_2)
        self.assertEqual(m.con_stage_2_expr, con_2)
        self.assertEqual(m.con_stage_2, None)
    
    def test_assign_constraint_con_1_expr(self):

        m = self.m

        def con1(m):
            return m.beta
        
        m.__con1 = Constraint(rule=con1)

        m.set_con_stage_1([m.__con1.expr])
        self.assertNotEqual(m.con_stage_1, [con1])
        self.assertEqual(m.con_stage_1, None)
    
    def test_assign_obj(self):

        m = self.m

        def obj(m):
            return m.alpha + m.beta
        
        objs = {
            'S1': obj,
            'S2': obj,
        }

        m.scenarios = self.scenarios
        
        m.set_obj(objs)

        self.assertEqual(m.obj, objs)
        self.assertEqual(m.obj_expr, None)
    
    def test_assign_obj_expr(self):

        m = self.m

        def obj(m):
            return m.alpha + m.beta
        __obj = Objective(expr=obj)

        objs = {
            'S1': __obj,
            'S2': __obj,
        }
        
        m.scenarios = self.scenarios
        
        m.set_obj(objs)

        self.assertEqual(m.obj, None)
        self.assertEqual(m.obj_expr, objs)
