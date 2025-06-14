from unittest import TestCase
from src.models.lagrangean_model import LagrangeanModel


class TestLagrangeanModel(TestCase):

    @classmethod
    def setUpClass(cls):

        from NSPLIB.src.instances.nonlinear_2D import const_model
        sto_m = const_model()
        cls.m = LagrangeanModel.from_sto_m(sto_m)
        cls.sto_m = sto_m
    
    def test_build_lagrangean_subproblems(self):

        self.m._build_lagrangean_subproblems()
        pass

    def test_solve(self):
        self.m.build()
        self.m.solve(max_iter=50, max_time=3600)
        pass


#     def test_lagrangean_colortv(self):
#         from methods.lagrangean import lagrangean_colortv

#         test_instance = StochasticInstance(self.data, self.model_type)

#         # Y = [self.data["yl"], self.data["yu"]]
#         # Y = [6-1e-3, 6+1e-3]
#         # Y = [5, 7]
#         ystar = 6.531128874581544
#         Y = [ystar - 1e-3, ystar + 1e-3]

#         # construct Lagrangean subproblems
#         test_instance.const_lagrangean_sub()

#         # upper bounding
#         upper_bounding(test_instance, Y)

#         # check if attributes exist
#         self.assertIsNone(test_instance.lagrangean_cuts)

#         lagrangean_colortv(test_instance, Y, output_level=1, max_lag_iter=100, conv_tol=1e-6)
#         self.assertIsNotNone(test_instance.lagrangean_cuts)
#         # print(test_instance.lagrangean_cuts)
#         print(f"number of cuts: {len(test_instance.lagrangean_cuts[0]['s1'])}")

#         print(f"LBD: {test_instance.lower_bound:.2f}, UBD: {test_instance.upper_bound:.2f}")

#         print(f"diff: {test_instance.upper_bound - test_instance.lower_bound:.2e}")

#         # self.fail()

#     def test_solve_node_lagrangean_colortv(self):
#         from methods.lagrangean import solve_node_lagrange_colortv

#         # test_instance = StochasticInstance(self.data, self.model_type)
#         test_instance = StochasticInstance(data_nlp_3_scenario, "nlp")

#         # Y = [self.data["yl"], self.data["yu"]]
#         # ystar = 6.531128874581544
#         ystar = 10
#         eps = 1
#         Y = [ystar - eps, ystar + eps]

#         kw = {
#             "max_lag_iter": 10,
#             "output_level": 2,
#             "y_diff_tol": 0.01,
#             "conv_tol": 1e-7,
#             "solver": "BARON",
#             "only_cuts": True,
#             "rule": "midpoint"
#         }

#         # bounding at the given interval
#         solve_node_lagrange_colortv(test_instance, Y,
#                                     # output_level=1,
#                                     # max_lag_iter=100,
#                                     # conv_tol=1e-6
#                                     **kw
#                                     )

#         # # LBD <= UBD
#         # try:
#         #     self.assertLessEqual(test_instance.sg_lower_bounds, test_instance.upper_bound)
#         # except AssertionError:
#         #     self.assertAlmostEqual(test_instance.sg_lower_bounds, test_instance.upper_bound)

#         # print(f"number of cuts: {len(test_instance.lagrangean_cuts[0]['s1'])}")
#         #
#         # print(f"LBD: {test_instance.lower_bound:.2f}, UBD: {test_instance.upper_bound:.2f}")
#         #
#         # print(f"diff: {test_instance.upper_bound - test_instance.lower_bound:.2e}")
#         #
#         # self.fail()


# class TestLagrangeanCuts(TestCase):

#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.dataset = [data_2_scenario, data_3_scenario, data_nlp_2_scenario, data_nlp_3_scenario]
#         cls.model_types = ["lp", "lp", "nlp", "nlp"]

#         cls.data = data_nlp_3_scenario
#         cls.model_type = "nlp"

#     def test_assign_mu(self):
#         # initialize StochasticInstance
#         test_instance = StochasticInstance(self.data, self.model_type)
#         test_instance.const_lagrangean_sub()

#         # initialize LagrangeanMultipliers
#         cuts = LagrangeanMultipliers(test_instance)

#         # test zero-value cuts
#         cuts.assign_mu(test_instance)
#         for si in test_instance.scenario:
#             self.assertEqual(cuts.mu_set[si][-1], test_instance.lagrangean_sub_model[si].mu)

#         # test random set cuts
#         import random
#         for _ in range(10):
#             for si in test_instance.scenario:
#                 cuts.mu_set[si].append(random.random())

#             cuts.assign_mu(test_instance)

#             for si in test_instance.scenario:
#                 self.assertEqual(cuts.mu_set[si][-1], test_instance.lagrangean_sub_model[si].mu)

#     def test_record_best_cuts(self):
#         # initialize StochasticInstance
#         test_instance = StochasticInstance(self.data, self.model_type)
#         test_instance.const_lagrangean_sub()
#         # initialize LagrangeanMultipliers
#         cuts = LagrangeanMultipliers(test_instance)

#         # test random values, see if the best cuts updated accordingly
#         import random
#         for _ in range(10):
#             # append random value for mu_set and z_SL
#             for si in test_instance.scenario:
#                 cuts.mu_set[si].append(random.random())
#                 cuts.z_SL[si].append(random.random())

#             # calc and update lbd
#             lbd_tmp = sum(cuts.z_SL[si][-1] for si in cuts.scenario)
#             cuts.sg_lower_bounds.append(lbd_tmp)

#             # record best cuts
#             cuts.record_best_cuts()
#             lbd_best = sum(cuts.best_z_SL[si] for si in cuts.scenario)

#             # # output
#             # print(f"current lbd: {lbd_tmp:.2f}\tbest lbd: {lbd_best:.2f}")

#             # assert
#             self.assertLessEqual(lbd_tmp, lbd_best)

#         # self.fail()

#     def test_record_obj(self):
#         import pyomo.environ as pe

#         # test for rule "lagrangean"

#         for data, model_type in zip(self.dataset, self.model_types):
#             # initialize
#             test_instance = StochasticInstance(data, model_type)
#             test_instance.const_lagrangean_sub()
#             cuts = LagrangeanMultipliers(test_instance)

#             # solve Lagrangean models
#             for si in test_instance.scenario:
#                 test_instance.solve(test_instance.lagrangean_sub_model[si])

#             # record obj
#             cuts.update_sg_lbd(test_instance)

#             # compare values
#             for si in test_instance.scenario:
#                 self.assertEqual(cuts.z_SL[si][-1], pe.value(test_instance.lagrangean_sub_model[si].obj))

#             self.assertEqual(sum(cuts.z_SL[si][-1] for si in test_instance.scenario), cuts.sg_lower_bounds[-1])

#             self.assertEqual(test_instance.lower_bound, cuts.sg_lower_bounds[-1])

#         # test for rule "midpoint"
#         # TODO tested in lagrangean_colortv

#     def test_update_s(self):
#         import pyomo.environ as pe
#         # initialize
#         for data, model_type in zip(self.dataset, self.model_types):
#             test_instance = StochasticInstance(data, model_type)
#             test_instance.const_lagrangean_sub()
#             cuts = LagrangeanMultipliers(test_instance)
#             # upper bounding
#             upper_bounding(test_instance, y_interval=[0, 20])
#             # solve the Lagrangean subproblems once
#             for si in test_instance.scenario:
#                 test_instance.solve(test_instance.lagrangean_sub_model[si])

#             # test lagrangean
#             rule = "lagrangean"
#             # record obj and s
#             cuts.update_sg_lbd(test_instance, rule=rule)
#             cuts.update_s(test_instance, rule=rule)
#             # test values
#             for si in cuts.scenario:
#                 diff_y = (pe.value(test_instance.lagrangean_sub_model[cuts.scenario[0]].y)
#                           - pe.value(test_instance.lagrangean_sub_model[si].y))
#                 if si == cuts.scenario[0]:
#                     continue
#                 self.assertEqual(diff_y, cuts.s_set[si][-1])

#             # test midpoint
#             rule = "midpoint"
#             cuts = LagrangeanMultipliers(test_instance)
#             # record obj and s
#             cuts.update_sg_lbd(test_instance, rule=rule)
#             cuts.update_s(test_instance, rule=rule)
#             # test values
#             for si in cuts.scenario:
#                 diff_y = cuts.y_mid - pe.value(test_instance.lagrangean_sub_model[si].y)
#                 self.assertEqual(diff_y, cuts.s_set[si][-1])

#     def test_update_mu(self):
#         import pyomo.environ as pe
#         import numpy as np

#         for data, model_type in zip(self.dataset, self.model_types):
#             # initialize
#             test_instance = StochasticInstance(data, model_type)
#             test_instance.const_lagrangean_sub()
#             cuts = LagrangeanMultipliers(test_instance)
#             # upper bounding
#             upper_bounding(test_instance, y_interval=[0, 20])
#             # solve the Lagrangean subproblems once
#             for si in test_instance.scenario:
#                 test_instance.solve(test_instance.lagrangean_sub_model[si])

#             # test lagrangean
#             rule = "lagrangean"
#             # record obj and s
#             cuts.update_sg_lbd(test_instance, rule=rule)
#             cuts.update_s(test_instance, rule=rule)
#             # update mu
#             cuts.update_mu(test_instance, theta=1)
#             # expect zero sum
#             mu_sum = sum(cuts.mu_set[si][-1] for si in cuts.scenario)
#             self.assertAlmostEqual(mu_sum, 0)
#             # test direction
#             for si in cuts.scenario:
#                 dir_y = (pe.value(test_instance.lagrangean_sub_model[cuts.scenario[0]].y)
#                          - pe.value(test_instance.lagrangean_sub_model[si].y))
#                 dir_mu = cuts.mu_set[si][-1] - cuts.mu_set[si][-2]
#                 if si == cuts.scenario[0]:
#                     continue
#                 self.assertEqual(np.sign(dir_y), - np.sign(dir_mu))

#             # test midpoint
#             rule = "midpoint"
#             cuts = LagrangeanMultipliers(test_instance)
#             # record obj and s
#             cuts.update_sg_lbd(test_instance, rule=rule)
#             cuts.update_s(test_instance, rule=rule)
#             # update mu
#             cuts.update_mu(test_instance, theta=1, rule=rule)
#             # test direction
#             for si in cuts.scenario:
#                 dir_y = cuts.y_mid - pe.value(test_instance.lagrangean_sub_model[si].y)
#                 dir_mu = cuts.mu_set[si][-1] - cuts.mu_set[si][-2]
#                 self.assertEqual(np.sign(dir_y), - np.sign(dir_mu))

#     def test_trim_mu(self):
#         import random
#         # initialize
#         test_instance = StochasticInstance(self.data, self.model_type)
#         test_instance.const_lagrangean_sub()
#         cuts = LagrangeanMultipliers(test_instance)

#         for si in cuts.scenario:
#             for _ in range(2):
#                 cuts.mu_set[si].append(random.random())
#                 cuts.z_SL[si].append(random.random())

#         self.assertGreater(len(cuts.mu_set[cuts.scenario[0]]), len(cuts.z_SL[cuts.scenario[0]]))
#         cuts.trim_mu()
#         self.assertEqual(len(cuts.mu_set[cuts.scenario[0]]), len(cuts.z_SL[cuts.scenario[0]]))


# class TestCutsColorTV(TestCase):
#     def test_stepsize_update(self):
#         self.fail()

#     def test_update_mu_colortv(self):
#         self.fail()

