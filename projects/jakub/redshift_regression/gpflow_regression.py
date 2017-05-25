import argparse

import numpy as np
import pandas as pd
import GPflow


LABEL_COL = 4
INPUT_COLS = 7, 9, 11, 13, 15
INPUT_DIM = len(INPUT_COLS)
INPUT_ROW_VALID = lambda row: row[2] == "Galaxy"

DEFAULT_TRAINING_SAMPLES_NUM = 1000
DEFAULT_TESTING_SAMPLES_NUM = 1000


def compute_R_sq(y_pred, y):
    observed_mean = np.mean(y)
    ss_tot = (y - observed_mean).dot(y - observed_mean)

    residuals = y_pred - y
    ss_res = residuals.dot(residuals)

    return 1 - ss_res / ss_tot


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
    train_y = data[:train_samples_num][[y_col]].as_matrix()
    test_y = data[train_samples_num
                  :train_samples_num+test_samples_num][[y_col]].as_matrix()

    assert train_X.shape == (train_samples_num, len(x_cols))
    assert train_y.shape == (train_samples_num, 1)
    assert test_X.shape == (test_samples_num, len(x_cols))
    assert test_y.shape == (test_samples_num, 1)

    return train_X, train_y, test_X, test_y


def main():
    parser = argparse.ArgumentParser(
        description=('Perform GPflow regression on photometric '
                     'redshifts and report results.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_n', metavar='TRAIN_N', type=int,
                        default=DEFAULT_TRAINING_SAMPLES_NUM,
                        help='number of training samples')
    parser.add_argument('--test_n', metavar='TEST_N', type=int,
                        default=DEFAULT_TESTING_SAMPLES_NUM,
                        help='number of testing samples')
    parser.add_argument('path', type=str, help='data file')

    args = parser.parse_args()

    train_X, train_y, test_X, test_y = load_data(args.path,
                                                 args.train_n,
                                                 args.test_n)

    k = GPflow.kernels.Matern52(5)
    m = GPflow.gpr.GPR(train_X, train_y, k)

    # Fit.
    m.optimize()

    # Predict and get score.
    predicted_y, _ = m.predict_y(test_X)
    score = compute_R_sq(predicted_y[:,0], test_y[:,0])
    print('R^2 score: {}'.format(score))


if __name__ == '__main__':
    main()
