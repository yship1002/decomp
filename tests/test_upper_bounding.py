# from unittest import TestCase
# from model_const import StochasticInstance


# class Test(TestCase):
#     def test_avg_upper_bounding(self):
#         from methods.upper_bounding import avg_upper_bounding

#         test_instance = StochasticInstance(data_nlp_2_scenario, "nlp")
#         Y = [data_2_scenario["yl"], data_2_scenario["yu"]]

#         # solve the Lagrangean subproblems first
#         test_instance.const_lagrangean_sub()
#         for si in test_instance.scenario:
#             test_instance.solve(test_instance.lagrangean_sub_model[si])

#         avg_upper_bounding(test_instance, Y)

#     def test_random_upper_bounding(self):
#         from methods.upper_bounding import random_upper_bounding

#         test_instance = StochasticInstance(data_nlp_2_scenario, "nlp")
#         Y = [data_2_scenario["yl"], data_2_scenario["yu"]]

#         # solve the Lagrangean subproblems first
#         test_instance.const_lagrangean_sub()
#         for si in test_instance.scenario:
#             test_instance.solve(test_instance.lagrangean_sub_model[si])

#         random_upper_bounding(test_instance, Y)

#     def test_upper_bounding(self):
#         from methods.upper_bounding import upper_bounding

#         test_instance = StochasticInstance(data_nlp_2_scenario, "nlp")
#         Y = [data_2_scenario["yl"], data_2_scenario["yu"]]
#         upper_bounding(test_instance, Y)

#         print(f"UBD: {test_instance.upper_bound:.2f}")
