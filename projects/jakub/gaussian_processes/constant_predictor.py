import csv
import sys

import numpy as np


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

    mean = np.mean(y)

    # Assume this predictor returns mean at all times.
    # Now compute R^2 of the constant predictor.
    mean_observed = np.mean(test_y)
    residuals = test_y - mean

    ss_res = residuals.dot(residuals)
    ss_tot = (y - mean_observed).dot(y - mean_observed)

    print("Score: {}".format(1 - ss_res / ss_tot))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError()
    main(sys.argv[1])
