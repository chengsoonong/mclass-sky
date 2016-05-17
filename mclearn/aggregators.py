""" Rank aggregators. """

import itertools
import functools
import numpy as np


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

    points = {}

    # accumulate the points for each candidate
    for voter in voters:
        for idx, candidate in enumerate(voter):
            points[candidate] = points.get(candidate, 1) * (idx + 1)

    # sort the candidates and return the most popular one(s)
    rank = sorted(points, key=points.__getitem__, reverse=False)
    return rank[:n_candidates]


def schulze_method(voters, n_candidates):
    """ Schulze method of ordering candidates.

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

    points = {}
    candidates = set()

    for voter in voters:
        for idx_i in range(len(voter)):
            for idx_j in range(idx_i + 1, len(voter)):
                i = voter[idx_i]
                j = voter[idx_j]
                points[(i, j)] = points.get((i, j), 0) + 1

    # get the id of all candidates
    candidates = set(np.ravel(voters))

    # compute the strogest path using a variant of Floyd–Warshall algorithm
    strength = {}
    for i, j in itertools.product(candidates, repeat=2):
        if i != j:
            points.setdefault((i, j), 0)
            points.setdefault((j, i), 0)
            if points[(i, j)] > points[(j, i)]:
                strength[(i, j)] = points[(i, j)]
            else:
                strength[(i, j)] = 0

    # k is the expanding set of intermediate nodes in Floyd-Warshall
    for k, i, j in itertools.product(candidates, repeat=3):
        if (i != j) and (k != i) and (k != j):
            strength[(i, j)] = max(strength[(i, j)], min(strength[(i, k)], strength[(k, j)]))

    # Schulze method guarantees that there is no cycle, so sorting is well-defined
    compare_strength = lambda x, y: strength[(x, y)] - strength[(y, x)]
    rank = sorted(candidates, key=functools.cmp_to_key(compare_strength), reverse=True)
    return rank[:n_candidates]

