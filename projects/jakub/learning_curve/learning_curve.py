import json
import sys

import numpy as np
import sklearn.gaussian_process
import sklearn.kernel_approximation
import sklearn.linear_model

# Import splitter
sys.path.insert(1, '..')
sys.path.insert(2, '../appx_gaussian_processes')
import appx_gp
import splitter


TRAINING_SAMPLES_NUM = 1000000
TESTING_SAMPLES_NUM = 50000

MAX_SGD = 50000
MAX_GP = 2500
MAX_AGP = 50000
STEP = 100

ALPHA = .003
LENGTH_SCALE = 1
GAMMA = .5 / (LENGTH_SCALE ** 2)
COMPONENTS = 100


def approximate_kernel(train_X, test_X):
    sampler = sklearn.kernel_approximation.RBFSampler(gamma=GAMMA, n_components=COMPONENTS)
    sampler.fit(train_X)
    appx_train_X = sampler.transform(train_X)
    appx_test_X = sampler.transform(test_X)
    return appx_train_X, appx_test_X


def perform_sgd(train_X, train_y, test_X, test_y):
    sgd = sklearn.linear_model.SGDRegressor(alpha=ALPHA, n_iter=100, fit_intercept=False, shuffle=True)
    sgd.fit(train_X, train_y)
    return sgd.score(test_X, test_y)


def perform_agp(train_X, train_y, test_X, test_y):
    agp = appx_gp.AppxGaussianProcessRegressor(alpha=ALPHA)
    agp.fit(train_X, train_y)
    out = agp.score(test_X, test_y)
    print('Done one approximate GP')
    return out


def perform_gp(train_X, train_y, test_X, test_y):
    kernel = sklearn.gaussian_process.kernels.RBF(
            length_scale=LENGTH_SCALE)
    gp = sklearn.gaussian_process.GaussianProcessRegressor(
        kernel=kernel,
        alpha=ALPHA,
        copy_X_train=False)
    gp.fit(train_X, train_y)
    return gp.score(test_X, test_y)


def main(path_in, path_out):
    data = splitter.load(path_in)
    (train_X, train_y), (test_X, test_y) \
        = splitter.split(data, TRAINING_SAMPLES_NUM, TESTING_SAMPLES_NUM)

    gp_x = list(range(STEP, MAX_GP+1, STEP))
    gp_y = []
    for i, n in enumerate(gp_x):
        print('Starting GP', i + 1)
        gp_y.append(perform_gp(train_X[:n], train_y[:n], test_X, test_y))

    transformed_train_X, transformed_test_X \
        = approximate_kernel(train_X[:max(MAX_AGP,MAX_SGD)], test_X)

    sgd_x = list(range(STEP, MAX_SGD+1, STEP))
    sgd_y = []
    for i, n in enumerate(sgd_x):
        print('Starting SGD', i + 1)
        sgd_y.append(perform_sgd(transformed_train_X[:n], train_y[:n],
                                 transformed_test_X, test_y))

    agp_x = list(range(STEP, MAX_AGP+1, STEP))
    agp_y = []
    for i, n in enumerate(agp_x):
        print('Starting AGP', i + 1)
        agp_y.append(perform_agp(transformed_train_X[:n], train_y[:n],
                                 transformed_test_X, test_y))

    with open(path_out, 'w') as f:
        json.dump({
                'gp_x': gp_x,
                'gp_y': gp_y,
                'sgd_x': sgd_x,
                'sgd_y': sgd_y,
                'agp_x': agp_x,
                'agp_y': agp_y
            }, f)


if __name__ == '__main__':
    main(*sys.argv[1:3])
