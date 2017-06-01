import numpy as np
import pandas as pd

def load(
        path,
        train_n,
        text_n,
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

    # Filter out anything that is not a galaxy without loading the
    # whole file into memory.
    data = pd.concat(chunk[chunk[class_col] == class_val]
                     for chunk in data_iter)

    X_data = data[x_cols_l].as_matrix()
    y_data = data[y_col].as_matrix()
    assert X_data.shape[0] == y_data.shape[0] == data.shape[0]
    assert X_data.shape[1] == data.shape[1] - 2
    assert len(y_data.shape) == 1

    # Shuffle data
    indices = list(range(data.shape[0]))
    np.random.seed(seed=12)
    np.random.shuffle(indices)
    X_data = X_data[indices]
    y_data = y_data[indices]

    train_X = X_data[:train_n]
    test_X = X_data[train_n:train_n+text_n]
    train_y = y_data[:train_n]
    test_y = y_data[train_n:train_n+text_n]

    assert train_X.shape == (train_n, len(x_cols))
    assert train_y.shape == (train_n,)
    assert test_X.shape == (text_n, len(x_cols))
    assert test_y.shape == (text_n,)

    return (train_X, train_y), (test_X, test_y)


if __name__ == '__main__':
    # Tiny test
    import sys
    if len(sys.argv) == 2:
        data = load(sys.argv[1], 2, 1)
        print(data)
