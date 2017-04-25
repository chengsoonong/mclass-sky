import numpy as np
import sklearn.dummy
import sklearn.gaussian_process
import sklearn.linear_model


class TestRegressionRsq:
    @classmethod
    def setup_class(cls):
        cls.train_X = np.array(
            [[24.66923, 23.05412, 21.47531, 20.41323, 20.04133],
             [23.43426, 22.80689, 21.82671, 20.86374, 20.04716],
             [23.73097, 22.83451, 21.53023, 20.70763, 19.86484],
             [23.78691, 21.84179, 20.21817, 19.59451, 19.09531],
             [23.29902, 21.96874, 21.10892, 20.4743,  19.88   ]])
        cls.train_y = np.array(
            [0.6179363, 0.6413319, 0.6637507, 0.3900273, 0.6044989])

        cls.test_X = np.array(
            [[21.60204, 20.66409, 20.15956, 19.97882, 19.73534],
             [23.94391, 23.00767, 21.29661, 20.37376, 19.89558],
             [21.51426, 19.85917, 18.76376, 18.22525, 17.76311]])
        cls.test_y = np.array(
            [0.04829505, 0.5284622, 0.2145626])

    def compute_rsq(self, regressor):
        y_pred = regressor.predict(self.test_X)

        observed_mean = np.mean(self.test_y)
        ss_tot = (self.test_y - observed_mean).dot(self.test_y - observed_mean)

        residuals = y_pred - self.test_y
        ss_res = residuals.dot(residuals)

        return 1 - ss_res / ss_tot

    def perform(self, regressor):
        regressor.fit(self.train_X, self.train_y)
        skscore = regressor.score(self.test_X, self.test_y)
        myscore = self.compute_rsq(regressor)

        assert skscore / myscore < 1.0000000001
        assert myscore / skscore < 1.0000000001
        assert skscore * myscore >= 0

    def test_const(self):
        regressor = sklearn.dummy.DummyRegressor()
        self.perform(regressor)

    def test_gp(self):
        regressor = sklearn.gaussian_process.GaussianProcessRegressor()
        self.perform(regressor)

    def test_sgd(self):
        regressor = sklearn.linear_model.SGDRegressor()
        self.perform(regressor)
