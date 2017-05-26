import concurrent.futures
import json
import sys

import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.kernel_approximation


TRAINING_SAMPLES_NUM = 1000000
TESTING_SAMPLES_NUM = 500000

MAX_SGD = 5000
MAX_GP = 5000
STEP = 1000


def load_data(
        path,
        x_cols=('psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z'),
        y_col='redshift',
        class_col='class',
        class_val='Galaxy'):

    import pandas as pd

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

    train_X = data[:TRAINING_SAMPLES_NUM][x_cols_l].as_matrix()
    test_X = data[TRAINING_SAMPLES_NUM
                  :TRAINING_SAMPLES_NUM
                   +TESTING_SAMPLES_NUM][x_cols_l].as_matrix()
    train_y = data[:TRAINING_SAMPLES_NUM][y_col].as_matrix()
    test_y = data[TRAINING_SAMPLES_NUM
                  :TRAINING_SAMPLES_NUM
                   +TESTING_SAMPLES_NUM][y_col].as_matrix()

    assert train_X.shape == (TRAINING_SAMPLES_NUM, len(x_cols))
    assert train_y.shape == (TRAINING_SAMPLES_NUM,)
    assert test_X.shape == (TESTING_SAMPLES_NUM, len(x_cols))
    assert test_y.shape == (TESTING_SAMPLES_NUM,)

    return train_X, train_y, test_X, test_y


def preprocess_sgd(X):
    rbf_feature = sklearn.kernel_approximation.RBFSampler(
        gamma=1e-1,
        random_state=1)
    return rbf_feature.fit_transform(X)


def perform_sgd(train_X, train_y, test_X, test_y):
    sgd = sklearn.linear_model.SGDRegressor()
    sgd.fit(train_X, train_y)
    return sgd.score(test_X, test_y)


def perform_gp(train_X, train_y, test_X, test_y):
    kernel = sklearn.gaussian_process.kernels.RationalQuadratic()
    gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(train_X, train_y)
    return gp.score(test_X, test_y)


def main(path):
    train_X, train_y, test_X, test_y = load_data(path)
    sgd_train_X, sgd_test_X = preprocess_sgd(train_X), preprocess_sgd(test_X)

    sgd_x = list(range(STEP, MAX_SGD+1, STEP))
    gp_x = list(range(STEP, MAX_GP+1, STEP))

    sgd_futures = []
    gp_futures = []
    e = concurrent.futures.ThreadPoolExecutor()
    for n in sgd_x:
        sgd_futures.append(e.submit(perform_sgd,
                                    sgd_train_X[:n], train_y[:n],
                                    sgd_test_X, test_y))
    for n in gp_x:
        gp_futures.append(e.submit(perform_gp,
                                   train_X[:n], train_y[:n],
                                   test_X, test_y))

    concurrent.futures.wait(sgd_futures + gp_futures)

    sgd_y = [f.result() for f in sgd_futures]

    gp_y =  [f.result() for f in gp_futures]

    with open('out.json', 'r') as f:
        json.dump({
                'sgd_x': sgd_x,
                'sgd_y': sgd_y,
                'gp_x': gp_x,
                'gp_y': gp_y
            }, f)


if __name__ == '__main__':
    main(sys.argv[1])
