from unittest import TestCase
from src.utility.plot import find_largest_power_10_smaller_than_equal, find_smallest_power_10_larger_than_equal
# from model_const import StochasticInstance
# import matplotlib.pyplot as plt

class TestPlotUtilities(TestCase):

    def test_find_largest_power_10_smaller_than_1(self):

        n = 12
        res = find_largest_power_10_smaller_than_equal(n)
        self.assertEqual(res, 10)

    def test_find_largest_power_10_smaller_than_2(self):

        n = 1
        res = find_largest_power_10_smaller_than_equal(n)
        self.assertEqual(res, 1)

    def test_find_largest_power_10_smaller_than_3(self):

        n = 0.1
        res = find_largest_power_10_smaller_than_equal(n)
        self.assertAlmostEqual(res, 0.1)

    def test_find_largest_power_10_smaller_than_4(self):

        n = 0.0023
        res = find_largest_power_10_smaller_than_equal(n)
        self.assertAlmostEqual(res, 0.001)
    
    def test_find_largest_power_10_smaller_than_5(self):

        n = 1e-5
        res = find_largest_power_10_smaller_than_equal(n)
        self.assertAlmostEqual(res, n)
    
    def test_find_smallest_power_10_larger_than_1(self):

        n = 12
        res = find_smallest_power_10_larger_than_equal(n)
        self.assertEqual(res, 100)

    def test_find_smallest_power_10_larger_than_2(self):

        n = 1
        res = find_smallest_power_10_larger_than_equal(n)
        self.assertEqual(res, 1)

    def test_find_smallest_power_10_larger_than_3(self):

        n = 0.1
        res = find_smallest_power_10_larger_than_equal(n)
        self.assertAlmostEqual(res, 0.1)

    def test_find_smallest_power_10_larger_than_4(self):

        n = 0.0023
        res = find_smallest_power_10_larger_than_equal(n)
        self.assertAlmostEqual(res, 0.01)

    def test_find_largest_power_10_smaller_than_5(self):

        n = 1e-5
        res = find_smallest_power_10_larger_than_equal(n)
        self.assertAlmostEqual(res, n)


#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.instance_info = instance_info_power_fun_obj_extra_y
#         cls.savefig = False

#     def test_v_y_plot(self):
#         from plotting.plot import plot_v

#         test_instance = StochasticInstance(instance_info_nonsmooth_v0_abs)

#         # specify y_interval
#         y_interval = [5, 6]
#         plot_v(test_instance, y_interval=y_interval, content="separate")
#         plot_v(test_instance, y_interval=y_interval, content="original")

#         # specify resolution
#         plot_v(test_instance, res=20)

#         # plot w/o specifying ax
#         plot_v(test_instance, content="separate")

#         # plot w/ specifying ax
#         fig, ax = plt.subplots()
#         plot_v(test_instance, ax=ax)
#         fig.show()

#         # savefig
#         plot_v(test_instance, content="separate", savefig=True)

#     def test_v_bar_b_plot(self):
#         from plotting.plot import plot_vb

#         instance_list = [instance_info_nonlinear_2, instance_info_nonlinear_3]
#         for instance in instance_list:
#             test_instance = StochasticInstance(instance)

#             # raise error when scenario > 2
#             if len(test_instance.data["scenario"]) > 2:
#                 with self.assertRaises(ValueError):
#                     plot_vb(test_instance)
#             else:
#                 # plot w/o specifying canvas
#                 plot_vb(test_instance)

#                 # plot w/ specifying canvas
#                 fig, ax = plt.subplots()
#                 plot_vb(test_instance, ax=ax)
#                 fig.show()

#     def test_vis_lagrangean_cuts(self):
#         from plotting.plot import vis_lower_bounds

#         test_instance = StochasticInstance(instance_info_nonsmooth_v0_abs)

#         ystar = 10
#         eps = 1e-3
#         Y = [ystar - eps, ystar + eps]

#         alg_list = [
#             # "lagrangean_multiple",
#             "lagrangean_best",
#             # "midpoint_multiple",
#             # "midpoint_best",
#         ]

#         kw = {
#             "max_lag_iter": 50,
#             "output_level": 2,
#             "solver": None,
#             "conv_tol": 1e-7,
#             "only_cuts": True,
#             "savefig": self.savefig
#         }

#         for alg in alg_list:
#             vis_lower_bounds(test_instance, y_interval=Y, alg_type=alg, **kw)

#     def test_vis_overall_convergence(self):
#         from plotting.plot import vis_overall_convergence

#         kw = {
#             "max_lag_iter": 50,
#             "output_level": 2,
#             "savefig": self.savefig,
#             "only_cuts": False,
#             "eps_l": -3,
#             "eps_u": 0,
#             "eps_res": 4,
#         }

#         alg_list = [
#             # "cao_zavala",
#             # "lagrangean_multiple",
#             "lagrangean_best",
#             # "midpoint",
#             # "midpoint_best",
#         ]

#         for method in alg_list:
#             test_instance = StochasticInstance(instance_info_nonsmooth_v0_abs)

#             y_range = [
#                 10,
#                 # 15
#             ]

#             vis_overall_convergence(test_instance, y_range, alg_type=method, **kw,
#                                     add_optimal_y=False)

#     def test_calc_resolution(self):
#         from plotting.plot import calc_res

#         Y = [0, 20]
#         self.assertEqual(calc_res(Y), 41)

#         Y = [6.5, 6.6]
#         self.assertEqual(calc_res(Y), 10)

#         Y = [-2, -3]
#         with self.assertRaises(ValueError):
#             calc_res(Y)


#     def test_obtain_v_y_profile(self):
#         self.fail()

#     def test_convergence_plot(self):

#         from plotting.plot import plot_convergence

#         for metric_type in [
#             # "hausdorff",
#             "pointwise",
#         ]:
#             for data, mtype in zip(self.data_set, self.mtype_set):
#                 test_instance = StochasticInstance(data, mtype)

#                 y_range = [
#                     None,
#                     7.5,
#                     # 12.5,
#                 ]
#                 algs = [
#                     # "cao_zavala",
#                     # "lagrangean_multiple",
#                     # "lagrangean_best",
#                     # "midpoint_multiple",
#                     "midpoint_best",
#                     # "li_grossmann",
#                 ]
#                 row_num = len(y_range)
#                 col_num = len(algs)
#                 fig, axes = plt.subplots(row_num, col_num, figsize=(5 * col_num, 5 * row_num))

#                 kw = {
#                     "metric": metric_type,
#                     "max_lag_iter": 10,
#                     "output_level": 1,
#                     "eps_l": -2,
#                     "eps_u": -1,
#                     "eps_res": 3,
#                     "solver": None,
#                     "only_cuts": True,
#                 }

#                 for i, y_star in enumerate(y_range):
#                     for j, method in enumerate(algs):
#                         try:
#                             plot_convergence(test_instance, y_star=y_star, alg_type=method,
#                                              ax=axes.ravel()[col_num * i + j], **kw)
#                         except AttributeError:  # only one plot
#                             plot_convergence(test_instance, y_star=y_star, alg_type=method, ax=axes, **kw)
#                 fig.show()

#     def test_calc_hausdorff_metric(self):
#         from plotting.plot import calc_hausdorff
#         import time
#         import numpy as np
#         test_instance = StochasticInstance(data_nlp_3_scenario, "nlp")
#         y_star = 6.5
#         eps = 1e-1
#         alg_type = "lagrangean_best"

#         kw = {
#             "max_lag_iter": 10,
#             "output_level": 0,
#             "y_diff_tol": 0.01,
#             "solver": "BARON",
#             "only_cuts": True,
#         }

#         tol_set = np.logspace(-10, -5, 6)
#         iteration = 1
#         for rel_tol in tol_set:
#             start_time = time.time()
#             for _ in range(iteration):
#                 calc_hausdorff(test_instance, y_star, eps, alg_type, kw, rel_tol)
#                 test_instance.reset()
#             print(f'\t\trelative tolerance: {rel_tol:.0e}: \t{(time.time() - start_time) / iteration:.3f} s')

#     def test_obtain_v_min_cv(self):
#         self.fail()

#     def test_overall_cut(self):
#         self.fail()

#     def test_calc_pointwise_metric(self):
#         self.fail()

#     def test_max_v_y(self):
#         self.fail()

#     def test_draw_best_lagrangean_cuts(self):
#         self.fail()

#     def test_draw_lagrangean_cuts(self):
#         self.fail()

#     def test_vis_cao_zavala_lower_bounding(self):
#         from plotting.plot import draw_cao_zavala_lower_bounding

#         # eps = 1
#         # Y = [7.5 - eps, 7.5 + eps]
#         Y = [0, 20]

#         test_instance = StochasticInstance(self.data, self.model_type)
#         draw_cao_zavala_lower_bounding(test_instance, y_interval=Y)

#         self.fail()


# if __name__ == "__main__":
#     unittest.main()
