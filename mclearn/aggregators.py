""" Rank aggregators. """


import functools
import numpy as np
from .schulze import schulze_method


def borda_count(voters, n_candidates):
    """ Borda count rank aggregator.

        Parameters
        ----------
        voters : list-like
            A list of arrays where each array correponds to a voter's preference.

        n_candidates : int
            The number of best candidates to be selected at each iteration.

        Returns
        -------
        best_candidates : array
            The list of indices of the best candidates.
    """

    max_score = len(voters[0])
    points = {}

    # accumulate the points for each candidate
    for voter in voters:
        for idx, candidate in enumerate(voter):
            points[candidate] = points.get(candidate, 0) + max_score - idx

    # sort the candidates and return the most popular one(s)
    rank = sorted(points, key=points.__getitem__, reverse=True)
    return rank[:n_candidates]


def geometric_mean(voters, n_candidates):
    """ Geometric mean rank aggregator.

        Parameters
        ----------
        voters : list-like
            A list of arrays where each array correponds to a voter's preference.

        n_candidates : int
            The number of best candidates to be selected at each iteration.

        Returns
        -------
        best_candidates : array
            The list of indices of the best candidates.
    """

    max_score = len(voters[0])
    points = {}

    # accumulate the points for each candidate
    for voter in voters:
        for idx, candidate in enumerate(voter):
            points[candidate] = points.get(candidate, 1) * (max_score - idx)

    # sort the candidates and return the most popular one(s)
    rank = sorted(points, key=points.__getitem__, reverse=True)
    return rank[:n_candidates]
