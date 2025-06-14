from unittest import TestCase
from src.analyses.value_function import ValueFunction
from src.models.cz_model import CaoZavalaModel


class Test(TestCase):

    def test_value_function(self):
        ...
    
    def test_mesh_3D(self):

        # from NSPLIB.src.instances.tmp_nonlinear_3D import const_model
        from NSPLIB.src.instances.nonlinear_2D import const_model
        sto_m = const_model()
        m = CaoZavalaModel.from_sto_m(sto_m)
        m.build()

        v_f = ValueFunction(m)

        v_f.calc_2D(0, 1, step=1)
        v_f.plot_2D()
        pass
