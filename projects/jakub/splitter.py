import numpy as np
import pandas as pd

def load(path,
         x_cols=('psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z'),
         y_col='redshift',
         class_col='class',
         class_val='Galaxy'):

    # Cast x_cols to list so Pandas doesn't complain…
    x_cols_l = list(x_cols)

    if '.h5' in path or '.hdf' in path:
        # We have an HDF5 file
        data = pd.read_hdf(path)
        return data, x_cols_l, y_col

    else:
        # We have a CSV file

        data_iter = pd.read_csv(
            path,
            iterator=True,
            chunksize=100000,
            usecols=x_cols_l + [y_col, class_col])

        # Filter out anything that is not a galaxy without loading the
        # whole file into memory.
        data = pd.concat(chunk[chunk[class_col] == class_val]
                         for chunk in data_iter)

        return data[x_cols_l + [y_col]], x_cols_l, y_col


def split(data, train_n, test_n):
    data, x_cols, y_col = data

    X_data = data[x_cols].as_matrix()
    y_data = data[y_col].as_matrix()
    assert X_data.shape[0] == y_data.shape[0] == data.shape[0]
    assert X_data.shape[1] == data.shape[1] - 1
    assert len(y_data.shape) == 1

    # Shuffle data
    indices = list(range(data.shape[0]))
    np.random.seed(seed=12)
    np.random.shuffle(indices)
    X_data = X_data[indices]
    y_data = y_data[indices]

    train_X = X_data[:train_n]
    test_X = X_data[train_n:train_n+test_n]
    train_y = y_data[:train_n]
    test_y = y_data[train_n:train_n+test_n]

    assert train_X.shape == (train_n, len(x_cols))
    assert train_y.shape == (train_n,)
    assert test_X.shape == (test_n, len(x_cols))
    assert test_y.shape == (test_n,)

    return (train_X, train_y), (test_X, test_y)


def save_as_hdf5(path, data):
    data.to_hdf(path, '👀')


if __name__ == '__main__':
    # Tiny test
    import sys
    if len(sys.argv) == 2:
        data = load(sys.argv[1])
        data = split(data, 2, 1)
        print(data)
    elif len(sys.argv) == 4 and sys.argv[2] == '--to-hdf5':
        data, _, _ = load(sys.argv[1])
        save_as_hdf5(sys.argv[3], data)
