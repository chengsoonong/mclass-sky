import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.gaussian_process.kernels


kernel = (sklearn.gaussian_process.kernels.ConstantKernel()
          + sklearn.gaussian_process.kernels.Matern(length_scale=2, nu=3/2)
          + sklearn.gaussian_process.kernels.WhiteKernel(noise_level=1))

LABEL_COL = 4
INPUT_COLS = 7, 9, 11, 13, 15
INPUT_DIM = len(INPUT_COLS)
INPUT_ROW_VALID = lambda row: row[2] == "Galaxy"
INPUT_SAMPLES_NUM = 1000
TESTING_SAMPLES_NUM = 1000
PLOT_SAMPLES = 1000


def take_samples(reader, num):
    X = np.empty((num, INPUT_DIM))
    y = np.empty((num,))

    i = 0
    for row in reader:
        if INPUT_ROW_VALID(row):
            y[i] = float(row[LABEL_COL])
            for j, col in enumerate(INPUT_COLS):
                X[i, j] = float(row[col])

            i += 1

            if i == num:
                break
    else:
        raise Exception("Not enough samples in file.")

    return X, y


def main(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip headers

        X, y = take_samples(reader, INPUT_SAMPLES_NUM)
        test_X, test_y = take_samples(reader, TESTING_SAMPLES_NUM)

    gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)

    if False:
        X_pred = np.empty((PRED_DATA, INPUT_DIM))
        X_pred[:, :4] = np.mean(X[:, :4], axis=0)
        X_pred[:, 4] = np.linspace(np.min(X[:, 4]), np.max(X[:, 4]), num=PRED_DATA)

        y_pred, sigmas = gp.predict(X_pred, return_std=True)
        plt.plot(X[:, 4], y, "ro", markersize=0.5)
        plt.errorbar(X_pred[:, 4], y_pred, yerr=sigmas, capsize=0)

        plt.show()

    print("Score: {}".format(gp.score(test_X, test_y)))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError()
    main(sys.argv[1])
