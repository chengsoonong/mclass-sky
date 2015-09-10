""" Useful general-purpose preprocessing functions. """

import gc
import os.path
import pandas as pd
import numpy as np
import nose.tools
from numpy.random import RandomState


def normalise_z(features):
    """ Normalise each feature to have zero mean and unit variance.
        
        Parameters
        ----------
        features : array, shape = [n_samples, n_features]
            Each row is a sample point and each column is a feature.
        
        Returns
        -------
        features_normalised : array, shape = [n_samples, n_features]
    """
       
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    return (features - mu) / sigma
    

def normalise_unit_var(features):
    """ Normalise each feature to have unit variance.
        
        Parameters
        ----------
        features : array, shape = [n_samples, n_features]
            Each row is a sample point and each column is a feature.
        
        Returns
        -------
        features_normalised : array, shape = [n_samples, n_features]
    """
    
    sigma = np.std(features, axis=0)
    return features / sigma
    
    
def normalise_01(features):
    """ Normalise each feature to unit interval.
        
        Parameters
        ----------
        features : array, shape = [n_samples, n_features]
            Each row is a sample point and each column is a feature.
        
        Returns
        -------
        features_normalised : array, shape = [n_samples, n_features]
    """
    
    minimum = np.min(features, axis=0)
    maximum = np.max(features, axis=0)
    return (features - minimum) / (maximum - minimum)


def _get_train_test_size(train_size, test_size, n_samples):
    """ Convert user train and test size inputs to integers. """

    if test_size is None and train_size is None:
        test_size = 0.3
        train_size = 1.0 - test_size

    if isinstance(test_size, float):
        test_size = np.round(test_size * n_samples).astype(int)

    if isinstance(train_size, float):
        train_size = np.round(train_size * n_samples).astype(int)

    if test_size is None:
        test_size = n_samples - train_size
    
    if train_size is None:
        train_size = n_samples - test_size

    return train_size, test_size


@nose.tools.nottest
def balanced_train_test_split(X, y, test_size=None, train_size=None, bootstrap=False,
                              random_state=None):
    """ Split the data into a balanced training set and test set of some given size.

        For a dataset with an unequal numer of samples in each class, one useful procedure
        is to split the data into a training and a test set in such a way that the classes
        are balanced.
        
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            Feature matrix.
        
        y : array, shape = [n_features]
            Target vector.

        test_size : float or int (default=0.3)
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
            to include in the test split. If int, represents the absolute number of test samples.
            If None, the value is automatically set to the complement of the train size.
            If train size is also None, test size is set to 0.3.
        
        train_size : float or int (default=1-test_size)
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
            to include in the train split. If int, represents the absolute number of train samples.
            If None, the value is automatically set to the complement of the test size.
            
        random_state : int, optional (default=None)
            Pseudo-random number generator state used for random sampling.
        
        Returns
        -------
        X_train : array
            The feature vectors (stored as columns) in the training set.
            
        X_test : array
            The feature vectors (stored as columns) in the test set.
            
        y_train : array
            The target vector in the training set.
            
        y_test : array
            The target vector in the test set.
    """
    
    # initialise the random number generator
    rng = RandomState(random_state)

    # make sure X and y are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    
    # get information about the class distribution
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    cls_count = np.bincount(y_indices)

    # get the training and test size
    train_size, test_size = _get_train_test_size(train_size, test_size, len(y))

    # number of samples in each class that is included in the training and test set
    n_train = np.round(train_size / n_classes).astype(int)
    n_test = np.round(test_size / n_classes).astype(int)
    n_total = n_train + n_test
    
    # make sure we have enough samples to create a balanced split
    min_count = min(cls_count)
    if min_count < (n_train + n_test) and not bootstrap:
        raise ValueError('The smallest class contains {} examples, which is not '
                         'enough to create a balanced split. Choose a smaller size '
                         'or enable bootstraping.'.format(min_count))
    
    # selected indices are stored here
    train = []
    test = []
    
    # get the desired sample from each class
    for i, cls in enumerate(classes):
        if bootstrap:
            shuffled = rng.choice(cls_count[i], n_total, replace=True)
        else:
            shuffled = rng.permutation(cls_count[i])
        
        cls_i = np.where(y == cls)[0][shuffled]
        train.extend(cls_i[:n_train])
        test.extend(cls_i[n_train:n_total])
        
    train = list(rng.permutation(train))
    test = list(rng.permutation(test))
    
    return X[train], X[test], y[train], y[test]


def csv_to_hdf(csv_path, no_files=1, hdf_path='store.h5', data_cols=None, expectedrows=7569900,
               min_itemsize=40, table_name='table'):
    """ Convert csv files to a HDF5 table.

        Parameters
        ----------
        csv_path : str
            The path of the source csv files.

        no_files : int
            The number of csv parts.

        hdf_path : str
            The path of the output.

        data_cols : array
            The names of the columns. Should be the same as the first line in the first csv file.

        expectedrows : int
            The expected number of rows in the HDF5 table.

        min_itemsize : int
            The minimum string size.

        table_name : str
            The name of the HDF5 table.
    """
    
    if os.path.isfile(hdf_path):
        print('HDF5 Table already exists. No changes were made.')
        return

    store = pd.HDFStore(hdf_path, complevel=9, complib='zlib', fletcher32=True)
    
    for i in np.arange(no_files):
        csv_file = csv_path.format(i)

        if i == 0:
            data = pd.io.api.read_csv(csv_file)
        else:
            data = pd.io.api.read_csv(csv_file, header=None, names=data_cols)
            
        store.append(table_name, data, index=False, expectedrows=expectedrows,
                     min_itemsize=min_itemsize)

        del data
        gc.collect()
        
    store.close()

