import os
import numpy as np
from pandas import read_csv
from mclearn.preprocessing import (
    normalise_z,
    normalise_unit_var,
    normalise_01,
    balanced_train_test_split,
    csv_to_hdf
)
from .datasets import Dataset


class TestNormalise:
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


class TestBalancedSplit:
    @classmethod
    def setup_class(cls):
        cls.sdss = Dataset('sdss_tiny')


    def test_balanced_split(self):
        X = self.sdss.features
        y = self.sdss.target
        X_train, X_test, y_train, y_test = balanced_train_test_split(
            X, y, train_size=72, test_size=45, random_state=13)
        
        # check that the training set is balanced
        classes, y_indices = np.unique(y_train, return_inverse=True)
        n_classes = len(classes)
        cls_count = np.bincount(y_indices)
        assert (np.sum(cls_count == 24) == n_classes)

        # check that the test set is balanced
        classes, y_indices = np.unique(y_test, return_inverse=True)
        n_classes = len(classes)
        cls_count = np.bincount(y_indices)
        assert (np.sum(cls_count == 15) == n_classes)


class TestHDF:
    @classmethod
    def setup_class(cls):
        cls.csv_path = 'mclearn/tests/data/sdss_tiny.csv'
        cls.hdf_path = 'mclearn/tests/data/sdss_tiny_nose_test.h5'
        cls.expectedrows = 328


    def test_csv_to_hdf(self):
        csv_to_hdf(self.csv_path, hdf_path=self.hdf_path, expectedrows=self.expectedrows)

        assert os.path.isfile(self.hdf_path)

    @classmethod
    def teardown_class(cls):
        if os.path.isfile(cls.hdf_path):
            os.remove(cls.hdf_path)

