from unittest import TestCase
from src.models.cz_model import CaoZavalaModel
from pyomo.environ import value


class TestCaoZavalaModel(TestCase):

    @classmethod
    def setUpClass(cls):

        from NSPLIB.src.instances.illustrative_examples.abs_func import const_model
        sto_m = const_model()
        cls.m = CaoZavalaModel.from_sto_m(sto_m)
        cls.sto_m = sto_m

    def test_check_binary_y(self):

        from NSPLIB.src.instances.pooling_contract_selection.pooling import const_model
        from src.models.cz_model import CaoZavalaModel

        sto_m = const_model()
        m = CaoZavalaModel.from_sto_m(sto_m)
        m.build()

        m._identify_binary_y()
        self.assertEqual(set(m.binary_y), set(
            [
                'lambd[1]', 'lambd[2]', 'lambd[3]', 'lambd[4]', 'lambd[5]',
                'theta[1]',
                'theta[2]',
                'theta[3]',
                'theta[4]',
            ]
        ))

    def test_fix_binary_y(self):

        binary_ys = ['lambd[1]', 'lambd[2]', 'lambd[3]', 'lambd[4]',
                     'lambd[5]', 'theta[1]', 'theta[2]', 'theta[3]', 'theta[4]']

        from NSPLIB.src.instances.pooling_contract_selection.pooling import const_model
        from src.models.cz_model import CaoZavalaModel

        sto_m = const_model()
        m = CaoZavalaModel.from_sto_m(sto_m)
        m.build()

        y_val = {v: 1 for v in binary_ys}

        m.fix_binary_y(y_val)

        # y values are stored
        self.assertEqual(y_val, m.binary_y_value)

        # these variables are excluded from y-set
        for v in binary_ys:
            self.assertNotIn(v, m.y_set)

        # these variables are excluded from y-bound
        for v in binary_ys:
            self.assertNotIn(v, m.y_bound)
        
        # original model values have been fixed
        y = m.origin_model.y
        for y_idx in binary_ys:
            self.assertEqual(value(y[y_idx]), 1)
        
        # lower bounding model values have been fixed
        for s in m.scenarios:
            y = m.lbd_models[s].y
            for y_idx in binary_ys:
                self.assertEqual(value(y[y_idx]), 1)

    def test_build_from_sto_m(self):

        m = self.m
        sto_m = self.sto_m

        scenarios = ['s1', 's2']
        y_set = [0]
        x_set = [0]

        Y = {
            0: [-10, 10],
        }

        X = {
            's1': {
                0: [-10, 10]
            },
            's2': {
                0: [-10, 10]
            }
        }

        self.assertEqual(m.scenarios, scenarios)
        self.assertEqual(m.y_set, y_set)
        self.assertEqual(m.x_set, x_set)
        self.assertEqual(m.y_bound, Y)
        self.assertEqual(m.x_bound, X)
        self.assertEqual(m.obj, sto_m.obj)
        self.assertEqual(m.con_stage_1, sto_m.con_stage_1)
        self.assertEqual(m.con_stage_2, sto_m.con_stage_2)

    def test_initialize(self):
        m = self.m
        m.build()
        m._initialize()

        root_node = m.node_list.get_node(0)
        self.assertEqual(root_node.bound, m.y_bound)

    def test_branch_1D(self):

        # Y = {
        #     0: [-10, 10],
        # }

        m = self.m
        m.build()
        m._initialize()

        y1_idx, y2_idx = m._branch(m.node_list.get_node(0))

        node_1 = m.node_list.get_node(y1_idx)
        node_2 = m.node_list.get_node(y2_idx)

        self.assertEqual(node_1.bound, {0: [-10, 0]})
        self.assertEqual(node_2.bound, {0: [0, 10]})

    def test_branch_2D(self):

        from NSPLIB.src.instances.illustrative_examples.nonlinear_2D import const_model
        sto_m = const_model()
        m = CaoZavalaModel.from_sto_m(sto_m)

        # Y = {
        #     0: [0, 20],
        #     1: [0, 20]
        # }

        Y1 = {
            0: [0, 10],
            1: [0, 20],
        }

        Y2 = {
            0: [10, 20],
            1: [0, 20],
        }

        m.build()
        m._initialize()

        y1_idx, y2_idx = m._branch(m.node_list.get_node(0))

        node_1 = m.node_list.get_node(y1_idx)
        node_2 = m.node_list.get_node(y2_idx)

        self.assertEqual(node_1.bound, Y1)
        self.assertEqual(node_2.bound, Y2)

        Y11 = {
            0: [0, 10],
            1: [0, 10],
        }

        Y12 = {
            0: [0, 10],
            1: [10, 20],
        }
        y11_idx, y12_idx = m._branch(m.node_list.get_node(y1_idx))

        node_11 = m.node_list.get_node(y11_idx)
        node_12 = m.node_list.get_node(y12_idx)

        self.assertEqual(node_11.bound, Y11)
        self.assertEqual(node_12.bound, Y12)


class TestCaoZavalaModelFromPyomo(TestCase):

    def test_build_from_pyomo(self):
        from NSPLIB.src.instances.pooling_contract_selection.pooling import const_model

        sto_m = const_model()
        m = CaoZavalaModel.from_sto_m(sto_m)

    def test_build_lbd_models_from_pyomo(self):
        from NSPLIB.src.instances.pooling_contract_selection.pooling import const_model

        sto_m = const_model()
        m = CaoZavalaModel.from_sto_m(sto_m)
        m.build()
