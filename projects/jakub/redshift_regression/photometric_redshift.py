import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.dummy
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.kernel_approximation


LABEL_COL = 4
INPUT_COLS = 7, 9, 11, 13, 15
INPUT_DIM = len(INPUT_COLS)
INPUT_ROW_VALID = lambda row: row[2] == "Galaxy"

DEFAULT_TRAINING_SAMPLES_NUM = 1000
DEFAULT_TESTING_SAMPLES_NUM = 1000


def load_gp_regressor():
    kernel = sklearn.gaussian_process.kernels.RationalQuadratic()
    return sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel)


def load_sgd_regressor():
    return sklearn.linear_model.SGDRegressor(
            # alpha=1e-2,
            # n_iter=100
        )

PREDICTOR_LOADERS = {'const': sklearn.dummy.DummyRegressor,
                     'GP': load_gp_regressor,
                     'SGD': load_sgd_regressor,
                     'linearSGD': load_sgd_regressor}

def preprocess_sgd(x):
    rbf_feature = sklearn.kernel_approximation.RBFSampler(
        gamma=1,
        random_state=1)
    x = rbf_feature.fit_transform(x)
    return x


NOOP = lambda x: x
PREPROCESSING = {'const': NOOP,
                 'GP': NOOP,
                 'SGD': preprocess_sgd,
                 'linearSGD': NOOP}

ADMIT_SIGMA = { 'GP' }


def take_samples(reader, num):
    X = np.empty((num, INPUT_DIM))
    y = np.empty((num,))

    for i, row in enumerate(filter(INPUT_ROW_VALID, reader)):
        if i == num:
            break

        y[i] = float(row[LABEL_COL])
        for j, col in enumerate(INPUT_COLS):
            X[i, j] = float(row[col])
    else:
        raise Exception("Not enough samples in file.")

    return X, y


def compute_R_sq(predictor, X, y):
    y_pred = predictor.predict(X)

    observed_mean = np.mean(y)
    ss_tot = (y - observed_mean).dot(y - observed_mean)

    residuals = y_pred - y
    ss_res = residuals.dot(residuals)

    return 1 - ss_res / ss_tot


def test_R_sq(score_a, predictor, X, y):
    score_b = compute_R_sq(predictor, X, y)
    if abs(score_b - score_a) < 1e-10:
        print('R^2 test passed.')
    else:
        print('R^2 test failed. Sklearn score: {}. '
              'Recomputed score: {}. Difference: {}.'.format(
                  score_a, score_b, abs(score_b - score_a)))


def plot(predictor, X, y, admits_sigma):
    if admits_sigma:
        y_pred, sigma = predictor.predict(X, return_std=True)
    else:
        y_pred = predictor.predict(X)

    assert y.shape == y_pred.shape  # Make sure sizes are the same
    assert len(y.shape) == 1  # Make sure both are vectors

    indices = np.argsort(y)
    y = y[indices]
    y_pred = y_pred[indices]
    if admits_sigma:
        sigma = sigma[indices]

    if admits_sigma:
        plt.errorbar(y, y_pred, yerr=sigma, fmt='x', ecolor='g')
    else:
        plt.scatter(y, y_pred, marker='x', s=10)
    plt.show()


def load_data(
        path,
        train_samples_num,
        test_samples_num,
        x_cols=('psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z'),
        y_col='redshift',
        class_col='class',
        class_val='Galaxy'):

    # Cast x_cols to list so Pandas doesn't complain…
    x_cols_l = list(x_cols)

    data_iter = pd.read_csv(
        path,
        iterator=True,
        chunksize=100000,
        usecols=x_cols_l + [y_col, class_col])

    # Filter out anything that is not a galaxy without loading the whole file into memory.
    data = pd.concat(chunk[chunk[class_col] == class_val]
                     for chunk in data_iter)

    train_X = data[:train_samples_num][x_cols_l].as_matrix()
    test_X = data[train_samples_num
                  :train_samples_num+test_samples_num][x_cols_l].as_matrix()
    train_y = data[:train_samples_num][y_col].as_matrix()
    test_y = data[train_samples_num
                  :train_samples_num+test_samples_num][y_col].as_matrix()

    assert train_X.shape == (train_samples_num, len(x_cols))
    assert train_y.shape == (train_samples_num,)
    assert test_X.shape == (test_samples_num, len(x_cols))
    assert test_y.shape == (test_samples_num,)

    return train_X, train_y, test_X, test_y


def main():
    parser = argparse.ArgumentParser(
        description=('Perform regression on photometric '
                     'redshifts and report results.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('predictor', metavar='P',
                        choices=PREDICTOR_LOADERS.keys(),
                        help='predictor to use for the regression')
    parser.add_argument('--train_n', metavar='TRAIN_N', type=int,
                        default=DEFAULT_TRAINING_SAMPLES_NUM,
                        help='number of training samples')
    parser.add_argument('--test_n', metavar='TEST_N', type=int,
                        default=DEFAULT_TRAINING_SAMPLES_NUM,
                        help='number of testing samples')
    parser.add_argument('path', type=str, help='data file')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='plot result of the regression')
    parser.add_argument('-t', '--test', action='store_true',
                        help='perform tests of R^2 values')
    parser.add_argument('-d', '--diffs', action='store_true',
                        help='investigate differences')
    args = parser.parse_args()

    predictor = PREDICTOR_LOADERS[args.predictor]()
    preprocessor = PREPROCESSING[args.predictor]

    train_X, train_y, test_X, test_y = load_data(args.path,
                                                 args.train_n,
                                                 args.test_n)

    # Add differences if wanted.
    if args.diffs:
        diffs_train_X = np.empty((train_X.shape[0], train_X.shape[1] - 1))

        for i in range(train_X.shape[1] - 1):
            # print(train_X[:,i])
            diffs_train_X[:,i] = train_X[:,i] - train_X[:,i+1]

        train_X = np.concatenate((train_X, diffs_train_X), axis=1)

        diffs_text_X = np.empty((test_X.shape[0], test_X.shape[1] - 1))

        for i in range(test_X.shape[1] - 1):
            # print(test_X[:,i])
            diffs_text_X[:,i] = test_X[:,i] - test_X[:,i+1]

        test_X = np.concatenate((test_X, diffs_text_X), axis=1)

    # Fit.
    train_X = preprocessor(train_X)
    predictor.fit(train_X, train_y)

    # Predict and get score.
    test_X = preprocessor(test_X)
    score = predictor.score(test_X, test_y)
    print('R^2 score: {}'.format(score))

    if args.test:
        test_R_sq(score, predictor, test_X, test_y)

    if args.plot:
        plot(predictor, test_X, test_y, args.predictor in ADMIT_SIGMA)


if __name__ == '__main__':
    main()
