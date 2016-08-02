import functools
import numpy as np


def _aggregate_votes_schulze(voters, n_best_candidates):
    """ Aggregate the ranks from the arms using the Schulze method. """
    candidates = np.unique(voters)
    n_candidates = len(candidates)

    # map candidates into integers from 0 to (n_candidates - 1)
    mapped_voters = np.digitize(voters, candidates, right=True)

    # construct the score matrix
    # pair (i, j) gets 1 point every time i is ranked higher than j
    points = np.zeros((n_candidates, n_candidates), dtype=int)
    for voter in mapped_voters:
        for idx_i in range(n_candidates):
            for idx_j in range(idx_i + 1, n_candidates):
                i = voter[idx_i]
                j = voter[idx_j]
                points[i, j] += 1

    # initialise the strength matrix as part of the Floyd-Warshall
    strength = np.zeros((n_candidates, n_candidates), dtype=int)
    for i in range(n_candidates):
        for j in range(n_candidates):
            if points[(i, j)] > points[j, i]:
                strength[i, j] = points[i, j]

    # k is the expanding set of intermediate nodes in Floyd-Warshall
    for k in range(n_candidates):
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j:
                    strength[i, j] = max(strength[i, j], min(strength[i, k], strength[k, j]))

    # Schulze method guarantees that there is no cycle, so sorting is well-defined
    compare_strength = lambda x, y: strength[x, y] - strength[y, x]
    rank = sorted(range(n_candidates),
                  key=functools.cmp_to_key(compare_strength),
                  reverse=True)
    return candidates[rank][:n_best_candidates]
