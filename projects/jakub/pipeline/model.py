import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.metrics

from rbfappxgp import RBFAppxGPRegressor


class RBFAppxGPWithCVRegressor(sklearn.base.BaseEstimator,
                               sklearn.base.RegressorMixin):
    def __init__(self, n_components=100, random_state=None, folds=5):
        self.rbf_gpr = RBFAppxGPRegressor()
        self.set_params(n_components=n_components,
                        random_state=random_state,
                        folds=folds)

    def get_params(self, deep=True):
        return dict(n_components=self.n_components,
                    random_state=self.random_state,
                    folds=self.folds)

    def set_params(self, **kwargs):
        if not kwargs.keys() <= {'n_components', 'random_state', 'folds'}:
            raise ValueError()
        if 'n_components' in kwargs:
            self.rbf_gpr.set_params(n_components=kwargs['n_components'])
            self.n_components = kwargs['n_components']
        if 'random_state' in kwargs:
            self.rbf_gpr.set_params(random_state=kwargs['random_state'])
            self.random_state = kwargs['random_state']
        if 'folds' in kwargs:
            self.folds = kwargs['folds']
        return self

    def fit(self, X, y=None, fit_mean=True,
                             fit_variance=False,
                             X_preprocessed=False,
                             reset=True):
        if X_preprocessed and reset:
            raise ValueError()
        if reset and y is None:
            raise ValueError()
        if fit_mean and y is None:
            raise ValueError()

        if reset:
            if self.random_state is not None:
                np.random.seed(self.random_state + 1)

            self.fit_y_transform(y)
            y = self._transform_y(y)

            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)

            X = X - self.X_mean
            X /= self.X_std

            ard = sklearn.linear_model.ARDRegression(
                n_iter=100,
                threshold_lambda=float('inf'),
                fit_intercept=False)
            ard.fit(X, y)

            self.gammas = np.sqrt(ard.lambda_)
            alpha = 1 / ard.alpha_

            X /= self.gammas
            self.rbf_gpr.set_params(alpha=alpha, gamma=1)
            self.rbf_gpr.fit(X, y, fit_mean=fit_mean,
                                   fit_variance=fit_variance,
                                   reset=True)
        else:
            if y is not None:
                y = self._transform_y(y)
            X = self._transform_X(X, X_preprocessed=X_preprocessed)
            self.rbf_gpr.fit(X, y,
                             fit_mean=fit_mean,
                             fit_variance=fit_variance,
                             X_preprocessed=X_preprocessed,
                             reset=False)

    def predict(self, X, return_mean=True,
                         return_std=False,
                         return_cov=False,
                         X_preprocessed=False):
        X = self._transform_X(X, X_preprocessed=X_preprocessed)
        prediction = self.rbf_gpr.predict(X, return_mean=return_mean,
                                             return_std=return_std,
                                             return_cov=return_cov,
                                             X_preprocessed=X_preprocessed)
        if return_mean:
            if return_std or return_cov:
                y, *rest = prediction
                y = self._detransform_y(y)
                return (y, *rest)
            else:
                y = prediction
                y = self._detransform_y(y)
                return y
        else:
            return prediction

    def _transform_X(self, X, X_preprocessed=False):
        if X_preprocessed:
            return X
        else:
            retval = X - self.X_mean
            retval /= self.X_std
            retval /= self.gammas
            return retval

    def _transform_y(self, y):
        y = y - self.y_mean
        y /= self.y_std
        return y

    def _detransform_y(self, y):
        y = y * self.y_std
        y += self.y_mean
        return y

    def preprocess_X(self, X):
        return self.rbf_gpr.preprocess_X(self._transform_X(X))

    def clone(self, clone_mean=True,
                    clone_variance=True,
                    deep_clone_transform=True):
        retval = RBFAppxGPWithCVRegressor.__new__(RBFAppxGPWithCVRegressor)
        retval.rbf_gpr = self.rbf_gpr.clone(
            clone_mean=clone_mean,
            clone_variance=clone_variance,
            deep_clone_transform=deep_clone_transform)
        self.folds = self.folds
        self.random_state = self.random_state
        return retval

    def fit_X_transform(self, X):
        self.rbf_gpr.fit_X_transform(X)
        self.X_transform_fitted = True

    def fit_y_transform(self, y):
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)

    def score(self, X, y, sample_weight=None, X_preprocessed=False):
        return sklearn.metrics.r2_score(
            y,
            self.predict(X, X_preprocessed=X_preprocessed),
            sample_weight=sample_weight,
            multioutput='variance_weighted')
