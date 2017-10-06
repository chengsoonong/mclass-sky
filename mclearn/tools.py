""" Some useful tools and helper functions. """

import pickle
import os
import sys
import pandas as pd
from urllib.request import urlopen


def results_exist(pickle_path):
    """ Check if the output file(s) already exist.

        Parameters
        ----------
        pickle_path : str or [str]
            The path(s) of the output files.

        Returns
        -------
        exist : boolean
            Indicate whether the all the output files exist.
    """

    # if only one file path is given
    if isinstance(pickle_path, str):
        return os.path.isfile(pickle_path)

    # if a list of paths is provided, check each path one-by-one
    for f in pickle_path:
        if not os.path.isfile(f):
            return False

    return True


def load_results(pickle_path):
    """ Load results from pickled files.

        Parameters
        ----------
        pickle_path : str or [str]
            The path(s) of the output files.

        Returns
        -------
        obj : object or [objects] or None
            The pickled object(s) are returned.
    """

    # make sure the the files actually exist
    if not results_exist(pickle_path):
        return None

    # if only one file path is given
    if isinstance(pickle_path, str):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    # if a list of paths is provided, return a list of objects
    results = []
    for p in pickle_path:
        with open(p, 'rb') as f:
            results.append(pickle.load(f))

    return results


def save_results(obj, pickle_path):
    """ Save results. """

    directory = os.path.dirname(pickle_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(obj, f, protocol=4)

    else:
        print('File already exists.')


def log(message, *messages, end='\n'):
    """ Flush messages immediately in Jupyter notebooks. """

    print(message, *messages, end=end)
    sys.stdout.flush()


def download_data(url,  dest, header=None, overwrite=True):
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if not os.path.exists(dest) or overwrite:
        with open(dest, 'wb') as f:
            if header:
                f.write(str.encode(header + '\n'))
            if type(url) is str:
                f.write(urlopen(url).read())
            elif type(url) is list:
                for u in url:
                    f.write(urlopen(u).read())


def fetch_data(url, dest, header=None, overwrite=True, process=True,
    process_fn=None, label=None, sep=','):
    """ Get data from the internet.

        Parameters
        ----------
        url : str
            The URL where the data can be downloaded.
        dest : str
            The destination path where the data will be stored.

        Returns
        -------
        data : array
            The downloaded data.
    """

    download_data(url, dest, header, overwrite)

    data = pd.read_csv(dest, sep=sep, na_values=['?'])
    if process:
        # id is not really a feature
        if 'id' in data.columns:
            data = data.drop('id', axis=1)
        if 'time' in data.columns:
            data = data.drop('time', axis=1)
        if 'placeholder' in data.columns:
            data = data.drop('placeholder', axis=1)

        # move target to first column
        data = data[['target']].join(data.drop('target', axis=1))

        if label:
            data.target = data.target.map(lambda x: label[x])

        # remove missing
        data.dropna(axis=0, how='any', inplace=True)

        if process_fn:
            data = process_fn(data)

        data.to_csv(dest, index=False, float_format='%.12g')

    return data
