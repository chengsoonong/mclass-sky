import numpy as np

import scipy.linalg
import sklearn.base
import sklearn.metrics


def _sym_pos_invert(a, overwrite_a=False):
    return scipy.linalg.solve(
        a,
        np.eye(a.shape[1]),
        overwrite_a=overwrite_a,
        overwrite_b=True,
        check_finite=False,
        assume_a='pos')


class AppxGaussianProcessRegressor(sklearn.base.BaseEstimator,
                                   sklearn.base.RegressorMixin):
    def __init__(self, alpha=1e-10):
        self.weights = None
        self.variance = None
        self.woodbury = None
        self.XTy_alph = None
        self.XTX_alph = None
        self.set_params(alpha=alpha)

    def get_params(self, deep=True):
        return {'alpha': self.alpha}

    def set_params(self, **kwargs):
        if not kwargs.keys() <= {'alpha'}:
            raise ValueError()
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        return self

    def _first_fit(self, X, y, fit_mean, fit_variance):
        n, f = X.shape
        diag_f = np.diag_indices(f)
        XTX_alph = X.T @ X
        XTX_alph /= self.alpha

        if n < f:
            # Fewer training points than features. Can optimise.
            diag_n = np.diag_indices(n)

            XXT = X @ X.T
            XXT[diag_n] += self.alpha

            if np.isscalar(XXT):
                woodbury = X.T @ X
                woodbury /= XXT
            else:
                XXT_inv = _sym_pos_invert(XXT, overwrite_a=True)
                woodbury = X.T @ XXT_inv @ X

            np.negative(woodbury, out=woodbury)
            woodbury[diag_f] += 1

        else:
            XTX_I = XTX_alph.copy()
            XTX_I[diag_f] += 1

            woodbury = _sym_pos_invert(XTX_I, overwrite_a=True)

        XTX_wdbry = XTX_alph @ woodbury

        if fit_mean:
            XTy_alph = X.T @ y
            XTy_alph /= self.alpha

            # In-place for efficiency
            weights = XTX_wdbry @ XTy_alph
            np.negative(weights, out=weights)
            weights += XTy_alph

            self.weights = weights
            self.XTy_alph = XTy_alph
        else:
            self.weights = None

        if fit_variance:
            variance = XTX_wdbry @ XTX_alph
            variance -= XTX_alph
            variance[diag_f] += 1

            self.variance = variance
        else:
            self.variance = None

        self.XTX_alph = XTX_alph
        self.woodbury = woodbury

    def _subsequent_fit(self, X, y, fit_mean, fit_variance):
        n, f = X.shape
        diag_f = np.diag_indices(f)
        XTX_new = X.T @ X
        XTX_new /= self.alpha

        XTX_alph = self.XTX_alph
        XTX_alph += XTX_new

        old_woodbury = self.woodbury

        if n < f:
            diag_n = np.diag_indices(n)

            XwXT = X @ old_woodbury @ X.T
            XwXT[diag_n] += self.alpha

            if np.isscalar(XwXT):
                woodbury = X.T @ X
                woodbury /= XwXT
            else:
                XwXT_inv = _sym_pos_invert(XwXT, overwrite_a=True)
                woodbury = X.T @ XwXT_inv @ X

            np.dot(old_woodbury, woodbury, out=woodbury)
            np.negative(woodbury, out=woodbury)
            woodbury[diag_f] += 1
            np.dot(woodbury, old_woodbury, out=woodbury)

        else:
            XTX_I = XTX_alph.copy()
            XTX_I[diag_f] += 1

            woodbury = _sym_pos_invert(XTX_I, overwrite_a=True)

        self.woodbury = woodbury
        XTX_wdbry = XTX_alph @ woodbury

        if fit_mean:
            if self.weights is None:
                raise ValueError()

            XTy_new = X.T @ y
            XTy_new /= self.alpha

            XTy_alph = self.XTy_alph
            XTy_alph += XTy_new

            # In-place for efficiency
            weights = self.weights
            weights = np.dot(XTX_wdbry, XTy_alph, out=weights)
            weights = np.negative(weights, out=weights)
            weights += XTy_alph
            self.weights = weights
        else:
            self.weights = None
            self.XTy_alph = None

        if fit_variance:
            if self.variance is None:
                raise ValueError()

            # In-place for efficiency
            variance = self.variance
            variance = np.dot(XTX_wdbry, XTX_alph, out=variance)
            variance -= XTX_alph
            variance[diag_f] += 1
            self.variance = variance
        else:
            self.variance = None

    def fit(self, X, y=None, fit_mean=True, fit_variance=False, reset=True):
        if not fit_mean and not fit_variance:
            raise ValueError()
        if fit_mean and y is None:
            raise ValueError()

        if reset:
            self._first_fit(X, y, fit_mean, fit_variance)
        else:
            self._subsequent_fit(X, y, fit_mean, fit_variance)

    def predict(self, X, return_mean=True, return_std=False, return_cov=False):
        if return_std and return_cov:
            raise ValueError()
        if not return_mean and not return_std and not return_cov:
            raise ValueError()

        retval = ()

        if return_mean:
            if self.weights is None:
                raise ValueError()

            mean = X @ self.weights
            retval += (mean,)

        if return_std:
            if self.variance is None:
                raise ValueError()

            # Compute diagonal of X @ self.variance @ X.T
            temp = X @ self.variance
            temp *= X

            std = np.sum(temp, axis=1)
            np.sqrt(std, out=std)

            retval += (std,)

        elif return_cov:
            if self.variance is None:
                raise ValueError()

            cov = X @ self.variance @ X.T
            retval += (cov,)

        if len(retval) == 1:
            return retval[0]
        else:
            return retval

    def clone(self, clone_mean=True, clone_variance=True):
        retval = AppxGaussianProcessRegressor(alpha=self.alpha)
        retval.XTX_alph = self.XTX_alph.copy()
        retval.woodbury = self.woodbury.copy()
        if clone_mean:
            retval.weights = self.weights.copy()
            retval.XTy_alph = self.XTy_alph.copy()
        if clone_variance:
            retval.variance = self.variance.copy()
        return retval

    def score(self, X, y, sample_weight=None, X_preprocessed=False):
        return sklearn.metrics.r2_score(
            y,
            self.predict(X, X_preprocessed=X_preprocessed),
            sample_weight=sample_weight,
            multioutput='variance_weighted')
