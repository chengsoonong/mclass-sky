""" Some useful tools and helper functions. """

import pickle
import os


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

    if not os.path.exists(pickle_path):
        os.makedirs(directory)

    if not os.path.exists(pickle_path):
        with open(pickle_path, 'wb') as f:
            pickle.dump(obj, f, protocol=4)

    else:
        print('File already exists.')