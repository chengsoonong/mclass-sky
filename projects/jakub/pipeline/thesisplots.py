import itertools
import pickle
import sys

sys.path.insert(1, '../thesis')
import thisisbullshit
thisisbullshit.do_this_bullshit(0.5, left=.12, bottom=.09, top=.1, ratio=1)


import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sklearn.gaussian_process
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing

import chris_error

sys.path.insert(1, '..')
import splitter


class KernelisedLinearRegressor():
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X, y):
        Phi = self.kernel(X)
        self.weights = scipy.linalg.solve(Phi, y, overwrite_a=True, check_finite=False)
        self.X_ = X

    def predict(self, X):
        Phi = self.kernel(X, self.X_)
        return Phi @ self.weights


class RegularisedLinearRegressor():
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        XT_y = X.T @ y
        XTX = X.T @ X
        XTX[np.diag_indices(XTX.shape[0])] += self.alpha
        woodbury_XT_y = scipy.linalg.solve(XTX, XT_y, overwrite_a=True, check_finite=False)
        self.weights = (XT_y - X.T @ (X @ woodbury_XT_y)) / self.alpha

    def predict(self, X):
        return X @ self.weights


class RegularisedKernelisedLinearRegressor():
    def __init__(self, kernel, alpha):
        self.kernel = kernel
        self.alpha = alpha

    def fit(self, X, y):
        Phi = self.kernel(X)
        Phi[np.diag_indices(Phi.shape[0])] += self.alpha
        self.weights = scipy.linalg.solve(Phi, y, overwrite_a=True, check_finite=False)
        self.X_ = X

    def predict(self, X):
        Phi = self.kernel(X, self.X_)
        return Phi @ self.weights



def get_colours(path):
    data = splitter.load(path)
    data, _ = splitter.split(data, data[0].shape[0], 0)
    X, y = data

    u, g, r, i, z = X.T
    u_g = u - g
    u_r = u - r
    r_i = r - i
    i_z = i - z

    X[:,0] = r
    X[:,1] = u_g
    X[:,2] = u_r
    X[:,3] = r_i
    X[:,4] = i_z
    data = X, y
    return data


def get_sdss_2dflens():
    sdss_path, dflens_path = sys.argv[1:]
    sdss_data = get_colours(sdss_path)
    dflens_data = get_colours(dflens_path)
    return sdss_data, dflens_data


def save_cache(name, *args):
    with open('{}.cache.pickle'.format(name), 'wb') as f:
        pickle.dump(args, f)


def load_cache(name):
    with open('{}.cache.pickle'.format(name), 'rb') as f:
        return pickle.load(f)


def plot_linreg_plain():
    try:
        means, stds = load_cache('linreg_plain')
    except FileNotFoundError:
        sdss_data, dflens_data = get_sdss_2dflens()

        means = []
        stds = []

        for data in sdss_data, dflens_data:
            X, y = data
            X = X[:50000]
            kf = sklearn.model_selection.KFold(n_splits=5,
                                            shuffle=True, random_state=1)
            scores = []
            for test_indices, train_indices in kf.split(y[:50000]):
                reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
                reg.fit(X[train_indices], y[train_indices])
                y_pred = reg.predict(X[test_indices])
                score = chris_error.sum_normalised_delta(y[test_indices], y_pred)
                scores.append(score)

            means.append(np.mean(scores))
            stds.append(np.std(scores))

        save_cache('linreg_plain', means, stds)

    for m, s in zip(means, stds):
        plt.errorbar(np.arange(1), [m], [s], fmt='o', lw=1)
    plt.xticks(np.arange(1), ['Linear regression'])
    plt.ylim(ymin=0)
    plt.legend(['SDSS', '2dFLenS'], bbox_to_anchor=(0.5, 1), loc='lower center', labelspacing=0)
    plt.ylabel('Mean $\delta$ error')
    plt.savefig('../thesis/linreg_plain.pdf')


def plot_linreg_polynomial():
    degrees = [None, 1, 2, 3]

    try:
        means, stds = load_cache('linreg_polynomial')
        datasets = [('sdss', None), ('dflens', None)]
    except FileNotFoundError:
        sdss_data, dflens_data = get_sdss_2dflens()
        datasets = [('sdss', sdss_data), ('dflens', dflens_data)]
        means = {}
        stds = {}

        for (dataset_name, data), degree in itertools.product(datasets, degrees):
            X, y = data
            X = X[:50000]
            if degree is not None:
                poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
                X = poly.fit_transform(X)
            kf = sklearn.model_selection.KFold(n_splits=5,
                                               shuffle=True, random_state=1)
            scores = []
            for test_indices, train_indices in kf.split(y[:50000]):
                reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
                reg.fit(X[train_indices], y[train_indices])
                y_pred = reg.predict(X[test_indices])
                score = chris_error.sum_normalised_delta(y[test_indices], y_pred)
                scores.append(score)

            means[dataset_name, degree] = np.mean(scores)
            stds[dataset_name, degree] = np.std(scores)

        save_cache('linreg_polynomial', means, stds)

    for dataset_name, _ in datasets:
        plt.errorbar(np.arange(len(degrees)),
                     [means[dataset_name, d] for d in degrees],
                     [stds[dataset_name, d] for d in degrees], fmt='o', lw=1)

    plt.xticks(np.arange(len(degrees)),
               ['Baseline' if d is None else '{}'.format(d) for d in degrees])
    plt.ylim(ymin=0, ymax=.06)
    plt.ylabel('Mean $\delta$ error')
    plt.xlabel('Degree')
    plt.margins(.2)
    plt.legend(['SDSS', '2dFLenS'], bbox_to_anchor=(0.5, 1), loc='lower center', labelspacing=0)
    plt.savefig('../thesis/linreg_polynomial.pdf')


def plot_linreg_kernelised():
    kernels = [('Baseline', None),
               ('Gaussian', sklearn.gaussian_process.kernels.RBF()),
               ('Rational Quadratic', sklearn.gaussian_process.kernels.RationalQuadratic()),
               ('Matérn', sklearn.gaussian_process.kernels.Matern()),
               ('Exponential', lambda *args, **kwargs: np.exp(sklearn.metrics.pairwise.euclidean_distances(*args, **kwargs)))]

    try:
        means, stds = load_cache('linreg_kernelised')
        datasets = [('SDSS', None), ('2dFLenS', None)]
    except FileNotFoundError:
        sdss_data, dflens_data = get_sdss_2dflens()
        datasets = [('SDSS', sdss_data), ('2dFLenS', dflens_data)]

        means = {}
        stds = {}

        for data_n, data in datasets:
            X, y = data
            X = X[:50000]
            X_ = X.copy()
            np.random.shuffle(X_)
            X_ -= X
            X_ *= X_
            d = np.sum(X_, axis=1)
            median_d = np.median(d)

            X /= median_d


            for kernel_n, kernel in kernels:
                if kernel is None:
                    poly = sklearn.preprocessing.PolynomialFeatures(degree=3)
                    X_ = poly.fit_transform(X)
                else:
                    X_ = X
                kf = sklearn.model_selection.KFold(n_splits=5,
                                                   shuffle=True, random_state=1)
                scores = []
                for test_indices, train_indices in kf.split(y[:50000]):
                    if kernel is None:
                        reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
                    else:
                        reg = KernelisedLinearRegressor(kernel=kernel)
                    reg.fit(X_[train_indices], y=y[train_indices])
                    y_pred = reg.predict(X_[test_indices])
                    score = chris_error.sum_normalised_delta(y[test_indices], y_pred)
                    scores.append(score)
                    break

                means[data_n, kernel_n] = np.mean(scores)
                stds[data_n, kernel_n] = np.std(scores)

        save_cache('linreg_kernelised', means, stds)

    for data_n, _ in datasets:
        plt.errorbar(np.arange(len(kernels)),
                     [means[data_n, k] for k, _ in kernels],
                     [stds[data_n, k] for k, _ in kernels],
                     fmt='o', lw=1)

    labels = {
        'Baseline': 'Baseline',
        'Matérn': "Mat\\'ern",
        'Rational Quadratic': 'Rat. Quad.',
        'Gaussian': 'Gaussian',
        'Exponential': 'Exponential'
    }

    plt.xticks(np.arange(len(kernels)), [labels[k] for k, _ in kernels], rotation=45)
    plt.margins(.2)
    plt.xlabel('Kernel')
    plt.ylabel(r'Mean $\delta$ error')
    plt.legend(['SDSS', '2dFLenS'], bbox_to_anchor=(0.5, 1), loc='lower center', labelspacing=0)
    plt.gca().set_yscale('log')
    plt.savefig('../thesis/linreg_kernelised.pdf')


def plot_linreg_kernelised_regularised():
    kernels = [('Baseline', None),
               ('Gaussian', sklearn.gaussian_process.kernels.RBF()),
               ('Rational Quadratic', sklearn.gaussian_process.kernels.RationalQuadratic()),
               ('Matérn', sklearn.gaussian_process.kernels.Matern()),
               ('Exponential', lambda *args, **kwargs: np.exp(sklearn.metrics.pairwise.euclidean_distances(*args, **kwargs)))
               ]

    try:
        means, stds = load_cache('linreg_kernelised_regularised')
        datasets = [('SDSS', None), ('2dFLenS', None)]
    except FileNotFoundError:
        sdss_data, dflens_data = get_sdss_2dflens()
        datasets = [('SDSS', sdss_data), ('2dFLenS', dflens_data)]

        means = {}
        stds = {}

        for data_n, data in datasets:
            X, y = data
            X = X[:50000]
            X_ = X.copy()
            np.random.shuffle(X_)
            X_ -= X
            X_ *= X_
            d = np.sum(X_, axis=1)
            median_d = np.median(d)

            X /= median_d


            for kernel_n, kernel in kernels:
                if kernel is None:
                    poly = sklearn.preprocessing.PolynomialFeatures(degree=3)
                    X_ = poly.fit_transform(X)
                else:
                    X_ = X
                kf = sklearn.model_selection.KFold(n_splits=5,
                                                   shuffle=True, random_state=1)
                scores = []
                for test_indices, train_indices in kf.split(y[:50000]):
                    if kernel is None:
                        reg = sklearn.linear_model.LinearRegression(fit_intercept=False)
                    else:
                        reg = RegularisedKernelisedLinearRegressor(kernel=kernel, alpha=1e-3)
                    reg.fit(X_[train_indices], y=y[train_indices])
                    y_pred = reg.predict(X_[test_indices])
                    score = chris_error.sum_normalised_delta(y[test_indices], y_pred)
                    scores.append(score)
                    break

                means[data_n, kernel_n] = np.mean(scores)
                stds[data_n, kernel_n] = np.std(scores)

        save_cache('linreg_kernelised_regularised', means, stds)

    for data_n, _ in datasets:
        plt.errorbar(np.arange(len(kernels)),
                     [means[data_n, k] for k, _ in kernels],
                     [stds[data_n, k] for k, _ in kernels],
                     fmt='o', lw=1)

    labels = {
        'Baseline': 'Baseline',
        'Matérn': "Mat\\'ern",
        'Rational Quadratic': 'Rat. Quad.',
        'Gaussian': 'Gaussian',
        'Exponential': 'Exponential'
    }

    plt.xticks(np.arange(len(kernels)), [labels[k] for k, _ in kernels], rotation=45)
    plt.margins(.2)
    plt.xlabel('Kernel')
    plt.ylabel(r'Mean $\delta$ error')
    plt.legend(['SDSS', '2dFLenS'], bbox_to_anchor=(0.5, 1), loc='lower center', labelspacing=0)
    plt.gca().set_yscale('log')
    plt.savefig('../thesis/linreg_kernelised_regularised.pdf')


def plot_linreg_kernel_appx():
    features_counts = [10, 100, 1000, 10000, None]

    try:
        means, stds = load_cache('linreg_kernel_appx')
        datasets = [('SDSS', None), ('2dFLenS', None)]
    except FileNotFoundError:
        sdss_data, dflens_data = get_sdss_2dflens()
        datasets = [('SDSS', sdss_data), ('2dFLenS', dflens_data)]

        means = {}
        stds = {}

        SPLITS = 5
        for data_n, data in datasets:
            print('Starting', data_n)
            X, y = data
            X = X[:50000]
            y = y[:50000]

            kf = sklearn.model_selection.KFold(n_splits=SPLITS,
                                               shuffle=True, random_state=1)
            alphas = []
            gammas = []

            for test_indices, train_indices in kf.split(y[:50000]):
                ard = sklearn.linear_model.ARDRegression(
                    n_iter=5,
                    threshold_lambda=float('inf'),
                    fit_intercept=False)
                ard.fit(X[train_indices[:1000]], y=y[train_indices[:1000]])

                alphas.append(1 / ard.alpha_)
                gammas.append(np.sqrt(ard.lambda_))
                break

            for f in features_counts:
                print('Starting', f)
                kf = sklearn.model_selection.KFold(n_splits=SPLITS,
                                                   shuffle=True, random_state=1)
                scores = []
                for (test_indices, train_indices), alpha, gamma in zip(kf.split(y[:50000]), alphas, gammas):
                    X_ = X / gamma
                    if f is None:
                        reg = RegularisedKernelisedLinearRegressor(kernel=sklearn.gaussian_process.kernels.RBF(), alpha=alpha)
                    else:
                        appx = sklearn.kernel_approximation.RBFSampler(n_components=f, random_state=1)
                        X_ = appx.fit_transform(X_)
                        reg = RegularisedLinearRegressor(alpha=alpha)
                    reg.fit(X_[train_indices], y=y[train_indices])
                    y_pred = reg.predict(X_[test_indices])
                    score = chris_error.sum_normalised_delta(y[test_indices], y_pred)
                    scores.append(score)
                    break

                means[data_n, f] = np.mean(scores)
                stds[data_n, f] = np.std(scores)

        save_cache('linreg_kernel_appx', means, stds)

    for data_n, _ in datasets:
        plt.errorbar(np.arange(len(features_counts)),
                     [means[data_n, f] for f in features_counts],
                     [stds[data_n, f] for f in features_counts],
                     fmt='o', lw=1)

    labels = {
        None: 'Exact',
        10: '10',
        100: '100',
        1000: '1K',
        10000: '10K'
    }

    plt.xticks(np.arange(len(features_counts)), [labels[f] for f in features_counts])
    plt.margins(.15)
    plt.xlabel(r'\# of features')
    plt.ylabel(r'Mean $\delta$ error')
    plt.legend(['SDSS', '2dFLenS'], bbox_to_anchor=(0.5, 1), loc='lower center', labelspacing=0)
    plt.ylim(ymin=0, ymax=.06)
    # plt.savefig('../thesis/linreg_kernel_appx.pdf')
    plt.show()


if __name__ == '__main__':
    plot_linreg_kernel_appx()
