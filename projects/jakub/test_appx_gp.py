import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process
import sklearn.kernel_approximation
import splitter

from appx_gaussian_processes import appx_gp


TRAINING_NUM = 1500
TESTING_NUM = 50000

ALPHA = .003
LENGTH_SCALE = 1
GAMMA = .5 / (LENGTH_SCALE ** 2)
COMPONENTS = 100


def interval_in_box_from_line(box, line):
    x_min, x_max, y_min, y_max = box
    m, b = line

    x_min_y = m * x_min + b
    x_max_y = m * x_max + b
    y_min_x = (y_min - b) / m
    y_max_x = (y_max - b) / m

    endpoints = set()
    if y_min <= x_min_y <= y_max:
        endpoints.add((x_min, x_min_y))
    if y_min <= x_max_y <= y_max:
        endpoints.add((x_max, x_max_y))
    if x_min <= y_min_x <= x_max:
        endpoints.add((y_min_x, y_min))
    if x_min <= y_max_x <= x_max:
        endpoints.add((y_max_x, y_max))

    return endpoints


def approximate_kernel(train_X, test_X):
    sampler = sklearn.kernel_approximation.RBFSampler(gamma=GAMMA, n_components=COMPONENTS)
    sampler.fit(train_X)
    appx_train_X = sampler.transform(train_X)
    appx_test_X = sampler.transform(test_X)
    return appx_train_X, appx_test_X


def main(path_in):
    print('Loading data...')
    data = splitter.load(path_in)
    (train_X, train_y), (test_X, test_y) = splitter.split(data, TRAINING_NUM,
                                                           TESTING_NUM)

    try:
        gp_sigmas = np.loadtxt('gp_preds.txt')
        assert gp_sigmas.shape == (TESTING_NUM,)
    except (FileNotFoundError, AssertionError):
        print('Fitting GP...')
        kernel = sklearn.gaussian_process.kernels.RBF(
            length_scale=LENGTH_SCALE)
        gp = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            alpha=ALPHA,
            copy_X_train=False)
        gp.fit(train_X, train_y)

        print('Predicting GP...')
        _, gp_sigmas = gp.predict(test_X, return_std=True)

        np.savetxt('gp_preds.txt', gp_sigmas)


    print('Approximating kernel...')
    appx_train_X, appx_test_X = approximate_kernel(train_X, test_X)

    print('Fitting approximate GP...')
    agp = appx_gp.AppxGaussianProcessRegressor(alpha=ALPHA)
    agp.fit(appx_train_X, train_y)

    print('Predicting approximate GP...')
    _, agp_sigmas = agp.predict(appx_test_X, return_std=True)

    print('Finding best fit...')
    best_fit = np.polyfit(gp_sigmas, agp_sigmas, 1)
    best_fit_box = (min(gp_sigmas), max(gp_sigmas),
                    min(agp_sigmas), max(agp_sigmas))
    best_fit_endpoints = interval_in_box_from_line(best_fit_box, best_fit)
    best_fit_xs, best_fit_ys = zip(*best_fit_endpoints)

    print('Plotting...')
    f = plt.figure()
    ax = f.add_subplot(111)
    sc = plt.scatter(gp_sigmas, agp_sigmas, s=.2, c=list(test_y))
    plt.plot(best_fit_xs, best_fit_ys, color='red', label='Linear fit')
    plt.title(r'$\gamma = {:.4},$ #components$= {}$'.format(GAMMA,
                                                            COMPONENTS))
    plt.xlabel('GP uncertainty')
    plt.ylabel('Approximate GP uncertainty')
    plt.text(.975, .1, '$y = {:.4}x {:+.4}$'.format(*best_fit),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform = ax.transAxes)
    colorbar = plt.colorbar(sc)
    colorbar.set_label('Redshift')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
