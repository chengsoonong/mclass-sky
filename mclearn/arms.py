""" Active learning suggestions. """

import logging
import numpy as np
from abc import ABC, abstractmethod
from numpy.random import RandomState
from scipy.stats import itemfreq

logger = logging.getLogger(__name__)

class Arm(ABC):
    def __init__(self, pool, labels, random_state=None):
        self.pool = pool
        self.labels = labels

        if type(random_state) is RandomState:
            self.seed = random_state
        else:
            self.seed = RandomState(random_state)

    @abstractmethod
    def select(self, candidate_mask, **kwargs):
        pass

    def _select_from_scores(self, candidate_mask, candidate_scores, n_best_candidates):
        pool_scores = np.full(len(candidate_mask), -np.inf)
        pool_scores[candidate_mask] = candidate_scores

        # make sure we don't return non-candidates
        n_best_candidates = min(n_best_candidates, len(candidate_scores))

        # sort from largest to smallest and pick the candidate(s) with the highest score(s)
        best_candidates = np.argsort(-pool_scores)[:n_best_candidates]
        return best_candidates


class RandomArm(Arm):
    def select(self, candidate_mask, predictions, n_best_candidates=1):
        candidate_indices = np.where(candidate_mask)[0]
        n_best_candidates = min(n_best_candidates, len(candidate_indices))
        random_candidates = seed.choice(candidate_indices, n_best_candidates, replace=False)
        return random_candidates


class WeightedArm(Arm):
    def __init__(self, pool, labels, random_state=None, similarity=None):
        super().__init__(pool, labels, random_state)
        self.similarity = similarity

    def _select_from_scores(self, candidate_mask, candidate_scores, n_best_candidates):
        # include density information, if the similarity matrix is supplied
        if self.similarity:
            density_weight = self.similarity[np.ix_(candidate_mask, self.labels.mask)]
            density_weight = np.mean(density_weight, axis=1)
            candidate_scores *= density_weight
        best_candidates = super()._select_from_scores(candidate_mask, candidate_scores, n_best_candidates)
        return best_candidates


class MarginArm(WeightedArm):
    def select(self, candidate_mask, predictions, n_best_candidates=1):
        # sort the probabilities from smallest to largest
        predictions = np.sort(predictions, axis=1)

        # compute the margin (difference between two largest probabilities)
        # the minus in front is there so that we can assign a higher score
        #   to those candidates with a smaller margin
        margin = 1 - np.abs(predictions[:, -1] - predictions[:, -2])

        best_candidates = self._select_from_scores(candidate_mask, margin, n_best_candidates)
        return best_candidates


class LeastConfidenceArm(WeightedArm):
    def select(self, candidate_mask, predictions, n_best_candidates=1):
        # extract the probability of the most likely label
        most_likely_probs = 1 - np.max(predictions, axis=1)

        best_candidates = self._select_from_scores(candidate_mask, most_likely_probs, n_best_candidates)
        return best_candidates


class EntropyArm(WeightedArm):
    def select(self, candidate_mask, predictions, n_best_candidates=1):
        # comptue Shannon entropy
        # in case of 0 * log(0), need to tell numpy to set it to zero
        entropy = -np.sum(np.nan_to_num(predictions * np.log(predictions)), axis=1)

        best_candidates = self._select_from_scores(candidate_mask, entropy, n_best_candidates)
        return best_candidates


class CommitteeArm(WeightedArm):
    self __init__(self, pool, labels, committee, n_committee_samples=300,
                  random_state=None, similarity=None)
        super().__init__(pool, labels, random_state, similarity)
        self.committee = committee
        self.n_committee_samples=n_committee_samples


def QBBMarginArm(CommitteeArm):
    def select(self, candidate_mask, predictions, n_best_candidates=1):
        # check that the max bagging sample is not too big
        self.committee.max_samples = min(self.n_committee_samples, np.sum(~self.labels.mask))

        # train the committee
        try:
            self.committee.fit(self.pool[~self.labels.mask], self.labels[~self.labels.mask])
        # the classifier will fail if there is only one class in the training set
        except ValueError:
            logger.info('Iteration {}: Class distribution is too skewed.'.format(
                         np.sum(~self.labels.mask)) +
                        'Falling back to the margin heuristic.')
            predictions = np.sort(predictions, axis=1)
            margin = 1 - np.abs(predictions[:, -1] - predictions[:, -2])
            best_candidates = self._select_from_scores(candidate_mask, margin)
            return best_candidates

        committee_predictions = self._predict(candidate_mask)

        # sort the probabilities from smallest to largest
        committee_predictions = np.sort(committee_predictions, axis=1)

        # compute the margin (difference between two largest probabilities)
        # the minus in front is there so that we can assign a higher score
        #   to those candidates with a smaller margin
        margin = 1 - np.abs(committee_predictions[:,-1] - committee_predictions[:,-2])

        best_candidates = self._select_from_scores(candidate_mask, margin, n_best_candidates)
        return best_candidates

    def _predict(self, candidate_mask):
        # predict
        n_samples = len(self.pool[candidate_mask])
        n_classes = len(self.committee.classes_)
        probs = np.zeros((n_samples, n_classes))
        class_freq = itemfreq(self.labels[~self.labels.mask])

        for member in self.committee.estimators_:
            member_prob = member.predict_proba(self.pool[candidate_mask])
            member_n_classes = member_prob.shape[1]

            if n_classes == member_n_classes:
                probs += member_prob

            else:
                member_classes = class_freq[:,1].argsort()[::-1]
                member_classes = member_classes[:member_n_classes]
                probs[:, member_classes] += member_prob[:, range(member_n_classes)]

        # average out the probabilities
        probs /= len(self.committee.estimators_)

        return probs


def QBBKLArm(CommitteeArm):
    def select(self, candidate_mask, predictions, n_best_candidates=1):
        # check that the max bagging sample is not too big
        self.committee.max_samples = min(self.n_committee_samples, np.sum(~self.labels.mask))

        # train the committee
        try:
            self.committee.fit(self.pool[~self.labels.mask], self.labels[~self.labels.mask])
        # the classifier will fail if there is only one class in the training set
        except ValueError:
            logger.info('Iteration {}: Class distribution is too skewed.'.format(
                         np.sum(~self.labels.mask)) +
                        'Falling back to passive learning.')
            candidate_indices = np.where(candidate_mask)[0]
            n_best_candidates = min(n_best_candidates, len(candidate_indices))
            random_candidates = seed.choice(candidate_indices, n_best_candidates, replace=False)
            return random_candidates

        avg_probs, prob_list = self._predict(candidate_mask)

        # compute the KL divergence
        avg_kl = np.zeros(avg_probs.shape[0])

        for p in prob_list:
            inner = np.nan_to_num(p * np.log(p / avg_probs))
            member_kl = np.sum(inner, axis=1)
            avg_kl += member_kl

        # average out the KL divergence
        avg_kl /= len(committee)

        best_candidates = self._select_from_scores(candidate_mask, margin, n_best_candidates)
        return best_candidates

    def _predict(self, candidate_mask):
        # predict
        n_samples = len(self.pool[candidate_mask])
        n_classes = len(self.committee.classes_)
        avg_probs = np.zeros((n_samples, n_classes))
        prob_list = []
        class_freq = itemfreq(self.labels[~self.labels.mask])

        for member in self.committee.estimators_:
            member_prob = member.predict_proba(self.pool[candidate_mask])
            member_n_classes = member_prob.shape[1]

            if n_classes == member_n_classes:
                avg_probs += member_prob
                prob_list.append(member_prob)

            else:
                member_classes = class_freq[:,1].argsort()[::-1]
                member_classes = member_classes[:member_n_classes]
                full_member_prob = np.zeros((n_samples, n_classes))
                full_member_prob[:, member_classes] += member_prob[:, range(member_n_classes)]
                avg_probs += full_member_prob
                prob_list.append(full_member_prob)

        # average out the probabilities
        avg_probs /= len(committee.estimators_)

        return (avg_probs, prob_list)
