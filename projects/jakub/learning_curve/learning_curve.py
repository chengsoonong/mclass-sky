import concurrent.futures
import json
import sys

import pandas as pd
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.kernel_approximation

# Import splitter
sys.path.insert(1, '..')
import splitter


TRAINING_SAMPLES_NUM = 1000000
TESTING_SAMPLES_NUM = 500000

MAX_SGD = 5000
MAX_GP = 1600
STEP = 100


def preprocess_sgd(train_X, train_y, test_X):
    kernel = sklearn.gaussian_process.kernels.Matern()
    subset = train_X[:1000]
    sgd_train_X = kernel(train_X, subset)
    sgd_test_X = kernel(test_X, subset)
    return sgd_train_X, sgd_test_X


    # rbf_feature = sklearn.kernel_approximation.RBFSampler(
    #     1 / (2 * ALPHA * ALPHA))
    # sgd_train_X = rbf_feature.fit_transform(train_X, train_y)
    # sgd_test_X = rbf_feature.fit_transform(test_X)
    # return sgd_train_X, sgd_test_X


def perform_sgd(train_X, train_y, test_X, test_y):
    sgd = sklearn.linear_model.SGDRegressor(alpha=0, n_iter=100)
    sgd.fit(train_X, train_y)
    out = sgd.score(test_X, test_y)
    print('Done one SGD')
    return out


def perform_gp(train_X, train_y, test_X, test_y):
    kernel = sklearn.gaussian_process.kernels.Matern()
    gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=0)
    gp.fit(train_X, train_y)
    out = gp.score(test_X, test_y)
    print('Done one GP')
    return out


def main(path):
    (train_X, train_y), (test_X, test_y) = splitter.load(
        path, TRAINING_SAMPLES_NUM, TESTING_SAMPLES_NUM)

    gp_x = list(range(STEP, MAX_GP+1, STEP))
    gp_y = [perform_gp(train_X[:n], train_y[:n], test_X, test_y)
            for n in gp_x]

    sgd_train_X, sgd_test_X = preprocess_sgd(train_X[:MAX_SGD], train_y[:MAX_SGD], test_X)

    sgd_x = list(range(STEP, MAX_SGD+1, STEP))
    sgd_y = [perform_sgd(sgd_train_X[:n], train_y[:n], sgd_test_X, test_y)
             for n in sgd_x]

    with open('gp2.json', 'w') as f:
        json.dump({
                # 'sgd_x': sgd_x,
                # 'sgd_y': sgd_y,
                'gp_x': gp_x,
                'gp_y': gp_y
            }, f)


if __name__ == '__main__':
    main(sys.argv[1])
