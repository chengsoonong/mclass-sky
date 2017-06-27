import numpy as np
import sklearn.base

class AppxGaussianProcessRegressor(sklearn.base.BaseEstimator,
                                   sklearn.base.RegressorMixin):
    """Approximate Gaussian process regression (GPR).

    Based on applying the Woodbury matrix identity to GPR according to
    https://github.com/chengsoonong/mclass-sky/issues/182

    Parameters
    ----------
    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations
        and reduce potential numerical issue during fitting. If an array is
        passed, it must have the same number of entries as the data used for
        fitting and is used as datapoint-dependent noise level. Note that this
        is equivalent to adding a WhiteKernel with c=alpha. Allowing to specify
        the noise level directly as a parameter is mainly for convenience and
        for consistency with Ridge.

    Attributes
    ----------
    alpha_ : number
        Dual coefficients of training data points in kernel space
    """

    def __init__(self, alpha=1e-10):
        self.alpha_ = alpha


    def fit(self, X, y):
        assert len(X.shape) == 2 and len(y.shape) == 1
        assert X.shape[0] == y.shape[0]

        XTX_alph = X.T @ X / self.alpha_
        XTy_alph = X.T @ y / self.alpha_
        eye = np.eye(XTX_alph.shape[0])

        woodbury = np.linalg.inv(eye + XTX_alph)

        self.uncertainty_ = eye - XTX_alph + XTX_alph @ woodbury @ XTX_alph
        self.weights_ = XTy_alph - XTX_alph @ woodbury @ XTy_alph


    def predict(self, X, return_std=False, return_cov=False):
        y_mean = X @ self.weights_

        if return_cov:
            y_covariance = X.T @ self.uncertainty_ @ X
            return y_mean, y_covariance

        elif return_std:
            y_var = np.zeros((X.shape[0],))

            for i, x in enumerate(X):
                y_var[i] = x.T @ self.uncertainty_ @ x

            return y_mean, np.sqrt(y_var)

        else:
            return y_mean


