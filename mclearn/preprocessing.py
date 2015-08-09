""" Useful general-purpose preprocessing functions. """

import gc
import pandas as pd
import numpy as np
from sklearn.cross_validation import ShuffleSplit


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


def draw_random_sample(data, train_size, test_size, random_state=None):
    """ Split the data into a train set and test set of a given size.
        
        Parameters
        ----------
        data : DataFrame, shape = [n_samples, n_features]
            Where each row is a sample point and each column is a feature.
            
        train_size : int
            Number of sample points in the training set.
            
        test_size : int
            Number of sample points in the test set.
            
        random_state : int, optional (default=None)
            Random seed.
        
        Returns
        -------
        combined_train_test : DataFrame
            Where each sample point (row) is indexed with either 'train' or 'test'.
    """
    
    m = len(data)    
    
    cv = ShuffleSplit(m, n_iter=1, train_size=train_size, test_size=test_size,
                      random_state=random_state)

    train, test = next(iter(cv))
    train_set = data.iloc[train]
    test_set = data.iloc[test]
    
    combined_train_test = pd.concat([train_set, test_set],
                                    keys=['train', 'test'], names=['set'])
    
    return combined_train_test
    
    
def balanced_train_test_split(data, features, target, train_size, test_size, random_state=None):
    """ Split the data into a balanced training set and test set of some given size.

        For a dataset with an unequal numer of samples in each class, one useful procedure
        is to split the data into a training and a test set in such a way that the classes
        are balanced.
        
        Parameters
        ----------
        data : DataFrame, shape = [n_samples, n_features]
            Where each row is a sample point and each column is a feature.
        
        features : array, shape = [n_features]
            The names of the columns in `data` that are used as feature vectors.
            
        target : str
            The name of the column in `data` that is used as the traget vector
        
        train_size : int
            Number of sample points from each class in the training set.
            
        test_size : int
            Number of sample points from each class in the test set.
            
        random_state : int, optional (default=None)
            Random seed.
        
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
    
    grouped = data.groupby(data[target])
    train_test = grouped.apply(lambda x: draw_random_sample(x, train_size, test_size, random_state))
    train_test = train_test.swaplevel(0, 1)
    
    X_train = train_test.loc['train', features].as_matrix()
    X_test =  train_test.loc['test', features].as_matrix()
    y_train = train_test.loc['train'][target].as_matrix()
    y_test =  train_test.loc['test'][target].as_matrix()
    
    return X_train, X_test, y_train, y_test


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
    
    store = pd.HDFStore(hdf_path, complevel=9, complib='zlib', fletcher32=True)
    
    for i in np.arange(no_files):
        csv_file = csv_path.format(i)

        if i == 0:
            data = pd.io.api.read_csv(csv_file)
        else:
            data = pd.io.api.read_csv(csv_file, header=None, names=data_cols)
            
        store.append(table_name, data, index=False, expectedrows=expectedrows, min_itemsize=min_itemsize)

        del data
        gc.collect()
        
    store.close()