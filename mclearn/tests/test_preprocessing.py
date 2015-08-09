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
        normalised = np.array([[-1.22474487,  1.09332714, -1.29777137],
                               [ 0.        ,  0.23017414,  0.16222142],
                               [ 1.22474487, -1.32350128,  1.13554995]])
        assert np.sum(normalise_z(self.data) - normalised) < 0.00001


    def test_normalise_unit_var(self):
        normalised = np.array([[0.40824829, 3.45261203, 0.        ],
                               [1.63299316, 2.58945902, 1.45999279],
                               [2.85773803, 1.03578361, 2.43332132]])
        assert np.sum(normalise_unit_var(self.data) - normalised) < 0.00001


    def test_normalise_01(self):
        normalised = np.array([[0. , 1.      , 0. ],
                               [0.5, 0.642857, 0.6],
                               [1. , 0.      , 1. ]])
        assert np.sum(normalise_01(self.data) - normalised) < 0.00001

