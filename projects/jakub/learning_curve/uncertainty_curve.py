import json
import sys

import numpy as np
import sklearn.gaussian_process

# Import splitter
sys.path.insert(1, '..')
import splitter


TRAINING_SAMPLES_NUM = 1000000
TESTING_SAMPLES_NUM = 1000

MAX_GP = 3000
STEP = 100

ALPHA = .002
LENGTH_SCALE = 1


def perform_gp(train_X, train_y, test_X):
    kernel = sklearn.gaussian_process.kernels.RBF(
            length_scale=LENGTH_SCALE)
    gp = sklearn.gaussian_process.GaussianProcessRegressor(
        kernel=kernel,
        alpha=ALPHA,
        copy_X_train=False)
    gp.fit(train_X, train_y)
    _, sigmas = gp.predict(test_X, return_std=True)
    return np.mean(sigmas)


def main(path_in, path_out):
    data = splitter.load(path_in)
    (train_X, train_y), (test_X, test_y) \
        = splitter.split(data, TRAINING_SAMPLES_NUM, TESTING_SAMPLES_NUM)

    gp_x = list(range(STEP, MAX_GP+1, STEP))
    gp_y = []
    for i, n in enumerate(gp_x):
        print('Starting GP', i + 1)
        gp_y.append(perform_gp(train_X[:n], train_y[:n], test_X))

    with open(path_out, 'w') as f:
        json.dump({
                'gp_x': gp_x,
                'gp_y': gp_y,
            }, f)


if __name__ == '__main__':
    main(*sys.argv[1:3])
