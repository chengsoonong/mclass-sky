import functools
import numpy as np


class SchulzeAggregator(ActiveAggregator):
    """ Schulze method of ordering candidates.

        Parameters
        ----------
        voters : ndarray
            A list of arrays where each array correponds to a voter's preference.

        n_candidates : int
            The number of best candidates to be selected at each iteration.

        Returns
        -------
        n_best_candidates : array
            The list of indices of the best candidates.
    """
    def _aggregate_votes(self, voters):
        candidates = np.unique(voters)
        total_candidates = len(candidates)

        # map candidates into integers from 0 to (total_candidates - 1)
        mapped_voters = np.digitize(voters, candidates, right=True)

        # construct the score matrix
        # pair (i, j) gets 1 point every time i is ranked higher than j
        points = np.zeros((total_candidates, total_candidates), dtype=int)
        for voter in mapped_voters:
            for idx_i in range(total_candidates):
                for idx_j in range(idx_i + 1, total_candidates):
                    i = voter[idx_i]
                    j = voter[idx_j]
                    points[i, j] += 1

        # initialise the strength matrix as part of the Floyd-Warshall
        strength = np.zeros((total_candidates, total_candidates), dtype=int)
        for i in range(total_candidates):
            for j in range(total_candidates):
                if points[(i, j)] > points[j, i]:
                    strength[i, j] = points[i, j]

        # k is the expanding set of intermediate nodes in Floyd-Warshall
        for k in range(total_candidates):
            for i in range(total_candidates):
                for j in range(total_candidates):
                    if i != j:
                        strength[i, j] = max(strength[i, j], min(strength[i, k], strength[k, j]))

        # Schulze method guarantees that there is no cycle, so sorting is well-defined
        compare_strength = lambda x, y: strength[x, y] - strength[y, x]
        rank = sorted(range(total_candidates), key=functools.cmp_to_key(compare_strength), reverse=True)
        return candidates[rank][:n_best_candidates]
