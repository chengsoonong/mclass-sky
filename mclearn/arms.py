""" Active learning suggestions.

    Module structure:
    - Arm
        - RandomArm
        - WeightedArm
            - MarginArm
            - ConfidenceArm
            - EntropyArm
            - CommitteeArm
                - QBBMarginArm
                - QBBKLArm
"""

# Author: Alasdair Tran
# License: BSD 3 clause

import logging
import numpy as np
from abc import ABC, abstractmethod
from numpy.random import RandomState
from scipy.stats import itemfreq

__all__ = ['RandomArm',
           'MarginArm',
           'ConfidenceArm',
           'EntropyArm',
           'QBBMarginArm',
           'QBBKLArm']

logger = logging.getLogger(__name__)

class Arm(ABC):
    """ Abstract base class for an active learning arm.

        This class cannot be used directly but instead serves as the base class for
        all active learning suggestions. Each suggestion implements the `select`
        method, which return the indices of the objects in the pool for labelling,
        based on a particular heuristic.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.
    """
    def __init__(self, pool, labels, random_state=None):
        self.pool = pool
        self.labels = labels

        if type(random_state) is RandomState:
            self.seed = random_state
        else:
            self.seed = RandomState(random_state)

    @abstractmethod
    def score(self, candidate_mask, predictions):
        """ Compute the score of each candidate. """
        pass

    def select(self, candidate_mask, predictions, n_best_candidates):
        """ Pick the candidates with the hihgest scores.

            Parameters
            ----------
            candidate_mask : numpy boolean array
                The boolean array that tells us which examples the arm is allowed to examine.

            predictions : numpy array
                Current class probabilities of the unlabelled candidates.

            n_best_candidates : int, optional (default=1)
                The number of best candidates to be returned.

            Returns
            -------
            best_candidates : int
                The indices of the best candidates.
        """
        scores = self.score(candidate_mask, predictions)
        best_candidates = self._select_from_scores(candidate_mask, scores, n_best_candidates)
        return best_candidates


    def _select_from_scores(self, candidate_mask, candidate_scores, n_best_candidates):
        """ Pick the candidates with the highest scores. """
        pool_scores = np.full(len(candidate_mask), -np.inf)
        pool_scores[candidate_mask] = candidate_scores

        # make sure we don't return non-candidates
        n_best_candidates = min(n_best_candidates, len(candidate_scores))

        # sort from largest to smallest and pick the candidate(s) with the highest score(s)
        best_candidates = np.argsort(-pool_scores)[:n_best_candidates]
        return best_candidates


class RandomArm(Arm):
    """ Pick random candidates from the unlabelled pool for querying (passive learning).

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.
    """
    def score(self, candidate_mask, predictions):
        """ Pick random candidates from the unlabelled pool.
            Parameters
            ----------
            candidate_mask : numpy boolean array
                The boolean array that tells us which examples the arm is allowed to examine.

            predictions : numpy array
                Current class probabilities of the unlabelled candidates.

            Returns
            -------
            scores : [float]
                The scores of the best candidates.
        """
        return self.seed.rand(len(predictions))


class WeightedArm(Arm):
    """ Abstract base class for a weighted active learning arm.

        This class cannot be used directly but instead serves as the base class for
        all weighted active learning suggestions. The weight uses a pre-computed
        similarity matrix to obtain the information density for each candidate.
        Candidates in a more dense region are given a higher weight.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        similarity : numpy array of shape [n_samples, n_samples], optional (default=None)
            A similarity matrix of all the examples in the pool. If not given,
            The information density will not be used.
    """
    def __init__(self, pool, labels, random_state=None, similarity=None):
        super().__init__(pool, labels, random_state)
        self.similarity = similarity

    def _select_from_scores(self, candidate_mask, candidate_scores, n_best_candidates):
        """ Pick the candidates with the highest scores, optionally weighted by the density. """
        if self.similarity is not None:
            density_weight = self.similarity[np.ix_(candidate_mask, self.labels.mask)]
            density_weight = np.mean(density_weight, axis=1)
            candidate_scores *= density_weight
        best_candidates = super()._select_from_scores(candidate_mask, candidate_scores,
                                                      n_best_candidates)
        return best_candidates


class MarginArm(WeightedArm):
    """ Suggests the candidate with the smallest margin.

        The margin is defined as the difference between the two largest values
        in the prediction vector.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        similarity : numpy array of shape [n_samples, n_samples], optional (default=None)
            A similarity matrix of all the examples in the pool. If not given,
            The information density will not be used.
    """
    def score(self, candidate_mask, predictions):
        """ Pick the candidates with the smallest margin.

            Parameters
            ----------
            candidate_mask : numpy boolean array
                The boolean array that tells us which examples the arm is allowed to examine.

            predictions : numpy array
                Current class probabilities of the unlabelled candidates.

            Returns
            -------
            scores : [float]
                The scores of the candidates.
        """
        # sort the probabilities from smallest to largest
        predictions = np.sort(predictions, axis=1)

        # compute the margin (difference between two largest probabilities)
        # the minus in front is there so that we can assign a higher score
        #   to those candidates with a smaller margin
        margin = 1 - np.abs(predictions[:, -1] - predictions[:, -2])

        return margin


class ConfidenceArm(WeightedArm):
    """ Suggests the candidate that we are least confident about its most likely labelling.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        similarity : numpy array of shape [n_samples, n_samples], optional (default=None)
            A similarity matrix of all the examples in the pool. If not given,
            The information density will not be used.
    """
    def score(self, candidate_mask, predictions):
        """ Pick the candidates with the smallest probability of the most likely label.

            Parameters
            ----------
            candidate_mask : numpy boolean array
                The boolean array that tells us which examples the arm is allowed to examine.

            predictions : numpy array
                Current class probabilities of the unlabelled candidates.

            Returns
            -------
            scores : [float]
                The scores of the candidates.
        """
        # extract the probability of the most likely label
        most_likely_probs = 1 - np.max(predictions, axis=1)

        return most_likely_probs


class EntropyArm(WeightedArm):
    """ Suggests the candidates whose prediction vectors display the greatest entropy.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        similarity : numpy array of shape [n_samples, n_samples], optional (default=None)
            A similarity matrix of all the examples in the pool. If not given,
            The information density will not be used.
    """
    def score(self, candidate_mask, predictions):
        """ Pick the candidates whose prediction vectors display the greatest entropy.

            Parameters
            ----------
            candidate_mask : numpy boolean array
                The boolean array that tells us which examples the arm is allowed to examine.

            predictions : numpy array
                Current class probabilities of the unlabelled candidates.

            Returns
            -------
            scores : [float]
                The scores of the candidates.
        """
        # comptue Shannon entropy
        # in case of 0 * log(0), need to tell numpy to set it to zero
        entropy = -np.sum(np.nan_to_num(predictions * np.log(predictions)), axis=1)

        return entropy


class CommitteeArm(WeightedArm):
    """ Abstract base class for a committee active learning arm.

        This class cannot be used directly but instead serves as the base class for
        all active learning suggestions that involve a committee.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        committee : BaggingClassifier object
            The committee should have the same interface as scikit-learn BaggingClassifier.

        n_committee_samples : int, optional (default=300)
            The maximum number of training examples that are given to each committee member
            during training.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        similarity : numpy array of shape [n_samples, n_samples], optional (default=None)
            A similarity matrix of all the examples in the pool. If not given,
            The information density will not be used.
    """
    def __init__(self, pool, labels, committee, n_committee_samples=300,
                  random_state=None, similarity=None):
        super().__init__(pool, labels, random_state, similarity)
        self.committee = committee
        self.n_committee_samples=n_committee_samples


class QBBMarginArm(CommitteeArm):
    """ Pick the candidates with the smallest average margins.

        We first use bagging to train a number of classifiers. The margin is then defined as
        the average difference between the two largest values in the prediction vector.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        committee : BaggingClassifier object
            The committee should have the same interface as scikit-learn BaggingClassifier.

        n_committee_samples : int, optional (default=300)
            The maximum number of training examples that are given to each committee member
            during training.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        similarity : numpy array of shape [n_samples, n_samples], optional (default=None)
            A similarity matrix of all the examples in the pool. If not given,
            The information density will not be used.
    """
    def score(self, candidate_mask, predictions):
        """ Compute the average margin of each candidate.

            Parameters
            ----------
            candidate_mask : numpy boolean array
                The boolean array that tells us which examples the arm is allowed to examine.

            predictions : numpy array
                Current class probabilities of the unlabelled candidates.

            Returns
            -------
            scores : [float]
                The scores of the candidates.
        """
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
            return self.seed.rand(len(predictions))

        committee_predictions = self._predict(candidate_mask)

        # sort the probabilities from smallest to largest
        committee_predictions = np.sort(committee_predictions, axis=1)

        # compute the margin (difference between two largest probabilities)
        # the minus in front is there so that we can assign a higher score
        #   to those candidates with a smaller margin
        margin = 1 - np.abs(committee_predictions[:,-1] - committee_predictions[:,-2])

        return margin

    def _predict(self, candidate_mask):
        """ Generate prediction vectors for the unlabelled candidates. """
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


class QBBKLArm(CommitteeArm):
    """ Pick the candidates with the largest average KL divergence from the mean.

        We first use bagging to train a number of classifiers. We then choose the candidate
        that has the largest Kullback–Leibler divergence from the average.

        Parameters
        ----------
        pool : numpy array of shape [n_samples, n_features]
            The feature matrix of all the examples (labelled and unlabelled).

        labels : numpy masked array of shape [n_samples].
            The missing entries of y corresponds to the unlabelled examples.

        committee : BaggingClassifier object
            The committee should have the same interface as scikit-learn BaggingClassifier.

        n_committee_samples : int, optional (default=300)
            The maximum number of training examples that are given to each committee member
            during training.

        random_state : int or RandomState object, optional (default=None)
            Provide a random seed if the results need to be reproducible.

        similarity : numpy array of shape [n_samples, n_samples], optional (default=None)
            A similarity matrix of all the examples in the pool. If not given,
            The information density will not be used.
    """
    def score(self, candidate_mask, predictions):
        """ Pick the candidates with the largest average KL divergence from the mean.

            Parameters
            ----------
            candidate_mask : numpy boolean array
                The boolean array that tells us which examples the arm is allowed to examine.

            predictions : numpy array
                Current class probabilities of the unlabelled candidates.

            Returns
            -------
            scores : [float]
                The scores of the candidates.
        """
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
            return self.seed.rand(len(predictions))

        avg_probs, prob_list = self._predict(candidate_mask)

        # compute the KL divergence
        avg_kl = np.zeros(avg_probs.shape[0])

        for p in prob_list:
            inner = np.nan_to_num(p * np.log(p / avg_probs))
            member_kl = np.sum(inner, axis=1)
            avg_kl += member_kl

        # average out the KL divergence
        avg_kl /= len(self.committee)

        return avg_kl

    def _predict(self, candidate_mask):
        """ Generate prediction vectors for the unlabelled candidates. """
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
        avg_probs /= len(self.committee.estimators_)

        return (avg_probs, prob_list)
