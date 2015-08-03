import numpy as np
from mclearn.preprocessing import (
    normalise_z,
    normalise_unit_var,
    normalise_01
)



class TestNormalise(object):
    @classmethod
    def setup_class(cls):
        cls.data = np.array([[1, 20, 0 ],
                         [4, 15, 6 ],
                         [7, 6 , 10]])
 

    def test_normalise_z(self):
        z_normalised = np.array([[-1.22474487,  1.09332714, -1.29777137],
                                 [ 0.        ,  0.23017414,  0.16222142],
                                 [ 1.22474487, -1.32350128,  1.13554995]])
        assert np.sum(normalise_z(self.data) - z_normalised) < 0.00001