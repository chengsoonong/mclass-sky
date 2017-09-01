import copy

import sklearn.base
import sklearn.kernel_approximation
import sklearn.metrics

from appxgp import AppxGaussianProcessRegressor


class RBFAppxGPRegressor(sklearn.base.BaseEstimator,
                         sklearn.base.RegressorMixin):
    def __init__(self, gamma=1.0,
                       n_components=100,
                       random_state=None,
                       alpha=1):
        self.gpr = AppxGaussianProcessRegressor()
        self.kernel_approx = sklearn.kernel_approximation.RBFSampler()
        self.fitted = False
        self.X_transform_fitted = False
        self.set_params(gamma=gamma, n_components=n_components,
                        random_state=random_state, alpha=alpha)

    def get_params(self, deep=True):
        return dict(
            gamma=self.kernel_approx.get_params()['gamma'],
            n_components=self.kernel_approx.get_params()['n_components'],
            random_state=self.kernel_approx.get_params()['random_state'],
            alpha=self.gpr.get_params()['alpha'])

    def set_params(self, **kwargs):
        if not kwargs.keys() <= {'gamma', 'n_components',
                                 'random_state', 'alpha'}:
            raise ValueError()

        if 'gamma' in kwargs:
            self.kernel_approx.set_params(gamma=kwargs['gamma'])
            self.X_transform_fitted = False
        if 'n_components' in kwargs:
            self.kernel_approx.set_params(n_components=kwargs['n_components'])
            self.X_transform_fitted = False
        if 'random_state' in kwargs:
            self.kernel_approx.set_params(random_state=kwargs['random_state'])
            self.X_transform_fitted = False
        if 'alpha' in kwargs:
            self.gpr.set_params(alpha=kwargs['alpha'])
        return self

    def fit(self, X, y=None, fit_mean=True,
                             fit_variance=False,
                             reset=True,
                             X_preprocessed=False):
        if fit_mean and y is None:
            raise ValueError()
        if X_preprocessed and not self.X_transform_fitted:
            raise ValueError()

        if not self.X_transform_fitted:
            self.fit_X_transform(X)

        Phi = self._transform_X(X, X_preprocessed=X_preprocessed)
        self.gpr.fit(Phi, y, fit_mean=fit_mean,
                             fit_variance=fit_variance,
                             reset=reset)
        self.fitted = True

    def predict(self, X, return_mean=True,
                         return_std=False,
                         return_cov=False,
                         X_preprocessed=False):
        Phi = self._transform_X(X, X_preprocessed=X_preprocessed)

        return self.gpr.predict(Phi, return_mean=return_mean,
                                     return_std=return_std,
                                     return_cov=return_cov)

    def clone(self, clone_mean=True,
                    clone_variance=True,
                    deep_clone_transform=True):
        retval = RBFAppxGPRegressor.__new__(RBFAppxGPRegressor)
        retval.gpr = self.gpr.clone(clone_mean=clone_mean,
                                    clone_variance=clone_variance)
        if deep_clone_transform:
            retval.kernel_approx = copy.deepcopy(self.kernel_approx)
        else:
            retval.kernel_approx = self.kernel_approx

        retval.fitted = self.fitted
        retval.X_transform_fitted = self.X_transform_fitted
        return retval

    def _transform_X(self, X, X_preprocessed=False):
        if X_preprocessed:
            return X
        else:
            return self.kernel_approx.transform(X)

    def preprocess_X(self, X):
        if not self.fitted:
            raise ValueError()

        return self._transform_X(X)

    def fit_X_transform(self, X):
        self.kernel_approx.fit(X)
        print('fitting approx with params', self.kernel_approx.get_params())
        self.X_transform_fitted = True

    def score(self, X, y, sample_weight=None, X_preprocessed=False):
        return sklearn.metrics.r2_score(
            y,
            self.predict(X, X_preprocessed=X_preprocessed),
            sample_weight=sample_weight,
            multioutput='variance_weighted')
