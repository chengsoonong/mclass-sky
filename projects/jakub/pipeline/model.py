import itertools

import numpy as np
import sklearn.base
import sklearn.kernel_approximation
import sklearn.model_selection


COMPONENTS = 100  # Higher values improve accuracy by better
                  # approximating the RBF kernel. However, the
                  # algorithm is O(n^3) in this component.

RANDOM_STATE = 1209333128  # Initial random state. Can be whatever.
                           # Explicitly set for reproducibility.


class AppxGPModel(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, **kwargs):
        self._alpha = None
        self._gamma = None
        self.set_params(**kwargs)

    def _find_normalization_constants(self, X, y):
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0, ddof=1)
        self.y_mean = y.mean()
        self.y_std = y.std(ddof=1)

    def _normalize_X(self, X):
        return (X - self.X_mean) / self.X_std

    def _normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def _denormalize_y(self, y):
        return y * self.y_std + self.y_mean

    def set_params(self, **kwargs):
        if 'alpha' in kwargs:
            self._alpha = kwargs['alpha']
        if 'gamma' in kwargs:
            self._gamma = kwargs['gamma']
        return self

    def get_params(self, deep=True):
        return {'alpha': self._alpha, 'gamma': self._gamma}

    def fit(self, X, y):
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] > 0

        self.d = X.shape[1]

        self._find_normalization_constants(X, y)
        X = self._normalize_X(X)
        y = self._normalize_y(y)

        self.kernel_appx = sklearn.kernel_approximation.RBFSampler(
            n_components=COMPONENTS,
            gamma=self._gamma,
            random_state=1209333128)
        self.kernel_appx.fit(X)

        Phi = self.kernel_appx.transform(X)

        self.PhiTPhi_alph = Phi.T @ Phi / self._alpha
        self.PhiTy_alph = Phi.T @ y / self._alpha
        eye = np.eye(self.PhiTPhi_alph.shape[0])

        woodbury = np.linalg.inv(eye + self.PhiTPhi_alph)

        self._uncertainty = (eye
                             - self.PhiTPhi_alph
                             + self.PhiTPhi_alph @ woodbury
                                                 @ self.PhiTPhi_alph)
        self._weights = (self.PhiTy_alph
                           - self.PhiTPhi_alph @ woodbury @ self.PhiTy_alph)

    def predict(self, X, return_std=False, return_cov=False):
        assert len(X.shape) == 2
        assert X.shape[1] == self.d

        X = self._normalize_X(X)
        Phi = self.kernel_appx.transform(X)

        y = Phi @ self._weights
        y = self._denormalize_y(y)

        if return_cov:
            y_cov = Phi.T @ self._uncertainty @ Phi
            return y, y_cov

        elif return_std:
            y_var = np.ndarray((X.shape[0],))

            for i, x in enumerate(Phi):
                y_var[i] = x @ self._uncertainty @ x

            y_std = np.sqrt(y_var)
            return y, y_std

        else:
            return y

    def recommend(self, X, n=1):
        assert len(X.shape) == 2
        assert X.shape[0] >= n > 0
        assert X.shape[1] == self.d

        X = self._normalize_X(X)
        Phi = self.kernel_appx.transform(X)

        indices = np.array(range(X.shape[0]))
        PhiTPhi_alph = self.PhiTPhi_alph
        eye = np.eye(PhiTPhi_alph.shape[0])
        retval = np.ndarray((n,), dtype=int)

        for i in range(n):
            woodbury = np.linalg.inv(eye + self.PhiTPhi_alph)
            uncertainty = (eye
                             - self.PhiTPhi_alph
                             + self.PhiTPhi_alph @ woodbury
                                                 @ self.PhiTPhi_alph)

            key = (lambda j: Phi[j] @ uncertainty @ Phi[j])
            best = max(range(Phi.shape[0]), key=key)
            phi = Phi[best]

            PhiTPhi_alph += phi.T @ phi / self._alpha

            retval[i] = indices[best]
            indices = np.delete(indices, best, 0)
            Phi = np.delete(Phi, best, 0)

        return retval

    def add_fit(self, X, y):
        try:
            assert len(X.shape) == 2
        except AssertionError:
            print(X.shape)
            raise
        assert len(y.shape) == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == self.d

        X = self._normalize_X(X)
        y = self._normalize_y(y)

        Phi = self.kernel_appx.transform(X)

        self.PhiTPhi_alph += Phi.T @ Phi / self._alpha
        self.PhiTy_alph += Phi.T @ y / self._alpha
        eye = np.eye(self.PhiTPhi_alph.shape[0])

        woodbury = np.linalg.inv(eye + self.PhiTPhi_alph)

        self._uncertainty = (eye
                             - self.PhiTPhi_alph
                             + self.PhiTPhi_alph @ woodbury
                                                 @ self.PhiTPhi_alph)
        self._weights = (self.PhiTy_alph
                           - self.PhiTPhi_alph @ woodbury @ self.PhiTy_alph)


class AppxGPModelWithCV(sklearn.base.BaseEstimator,
                        sklearn.base.RegressorMixin):
    def __init__(self,
                 ls_try_percentiles=(10, 25, 50, 75, 90),
                 try_alphas=(.1, .01, .001, .0001, .00001)):
        self._ls_try_percentiles = ls_try_percentiles
        self._try_alphas = try_alphas

    def fit(self, X, y):
        np.random.seed(RANDOM_STATE + 1)
        X_samples = np.random.choice(X.shape[0], (2, X.shape[0]))
        distances = np.linalg.norm(X[X_samples[0]] - X[X_samples[1]], axis=1)

        assert distances.shape == (X.shape[0],)

        length_scales = [np.percentile(distances, p)
                         for p in self._ls_try_percentiles]

        params = {'alpha': self._try_alphas, 'gamma': length_scales}
        clf = sklearn.model_selection.GridSearchCV(
            AppxGPModel(),
            params,
            cv=sklearn.model_selection.KFold(n_splits=5,
                                             shuffle=True,
                                             random_state=RANDOM_STATE + 2))
        clf.fit(X, y)
        self.regressor = clf.best_estimator_

    def predict(self, X):
        return self.regressor.predict(X)

    def recommend(self, X, n=1):
        return self.regressor.recommend(X, n)

    def add_fit(self, X, y):
        self.regressor.add_fit(X, y)
