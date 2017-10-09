""" Active learning experiment. """

# Author: Alasdair Tran
# License: BSD 3 clause

import os
import numpy as np
from time import time
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from mclearn.arms import RandomArm, MarginArm, ConfidenceArm, EntropyArm, QBBMarginArm, QBBKLArm
from mclearn.performance import mpba_score, micro_f1_score
from mclearn.policies import SingleSuggestion, ThompsonSampling, OCUCB, KLUCB, EXP3PP, ActiveAggregator


def save_results(dataset, policy, results):
    dir_path = os.path.join('results', dataset, policy)
    os.makedirs(dir_path, exist_ok=True)
    dump_path = os.path.join(dir_path, 'results.pkl')
    joblib.dump(results, dump_path)


def load_results(dataset, policy, measure=None, mean=False):
    dump_path = os.path.join('results', dataset, policy, 'results.pkl')
    try:
        results = joblib.load(dump_path)
        results = results[measure] if measure else results
        results = results.mean(axis=0) if mean else results
        return results
    except FileNotFoundError:
        return None


def sample_from_every_class(y, size, seed=None):
    """ Get a random sample, ensuring every class is represented.

        This helper function is useful when the sample size is small and
        we want to make sure that at least one sample from each class
        is included. This is required, for example, in logistic regression,
        where the classifier cannot handle classes where it has never seen
        any training examples.

        Params
        ------
        y : 1-dimensional numpy array
            The label/output array.
        size : int
            The desired number of samples.
        seed : RandomState object or None
            Provide a seed for reproducibility.

        Returns
        -------
        samples : numpy array of shape [size]
            The random samples.
    """
    if seed is None:
        seed = RandomState(1234)

    # Keep track of the classes which have not been sampled yet
    labels = np.unique(y)
    samples = []
    while len(samples) < size:
        idx = seed.choice(np.arange(len(y)))
        if len(labels) == 0 or y[idx] in labels:
            samples.append(idx)
            labels = np.delete(labels, np.argwhere(labels == y[idx]))
    return samples


class ActiveExperiment:
    """ Simulate an active learning experiment. """
    def __init__(self, X, y, dataset, policy_name, scale=True, n_iter=10, passive=True):
        seed = RandomState(1234)
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        self.X = StandardScaler().fit_transform(self.X) if scale else self.X
        self.policy_name = policy_name
        self.dataset = dataset
        self.passive = passive

        # estimate the kernel using the 90th percentile heuristic
        random_idx = seed.choice(X.shape[0], 1000)
        distances = pairwise_distances(self.X[random_idx], metric='l1')
        self.gamma = 1 / np.percentile(distances, 90)
        transformer = RBFSampler(gamma=self.gamma, random_state=seed, n_components=100)
        self.X_transformed = transformer.fit_transform(self.X)

        n_samples = self.X.shape[0]
        train_size = min(10000, int(0.7 * n_samples))
        test_size = min(20000, n_samples - train_size)
        self.kfold = StratifiedShuffleSplit(self.y, n_iter=n_iter, test_size=test_size,
                                            train_size=train_size, random_state=seed)

    def run_policies(self):
        start_time = time()
        outputs = Parallel(n_jobs=-1)(delayed(self._run_fold)(train_index, test_index)
                                      for train_index, test_index in self.kfold)
        end_time = time()

        # unpack results
        results = {}
        for key in outputs[0]:
            results[key] = [fold[key] for fold in outputs]
            results[key] = np.asarray(results[key])
            results['time'] = end_time - start_time
        save_name = self.policy_name if self.passive else self.policy_name + '-wop'
        save_results(self.dataset, save_name, results)

    def run_asymptote(self):
        results = {
            'asymptote_mpba': [],
            'asymptote_accuracy': [],
            'asymptote_f1': []}

        for train_index, test_index in self.kfold:
            X_train = self.X_transformed[train_index]
            X_test = self.X_transformed[test_index]
            y_train = self.y[train_index]
            y_test = self.y[test_index]
            n_classes = len(np.unique(y_test))
            seed = RandomState(1234)
            classifier = LogisticRegression(multi_class='ovr', penalty='l2', C=1000,
                                            random_state=seed, class_weight='balanced')
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            results['asymptote_mpba'].append(mpba_score(y_test, y_pred))
            results['asymptote_accuracy'].append(accuracy_score(y_test, y_pred))
            results['asymptote_f1'].append(micro_f1_score(y_test, y_pred, n_classes))

        for key in results:
            results[key] = np.array(results[key])

        save_results(self.dataset, 'asymptote', results)

    def _run_fold(self, train_index, test_index):
        # reset the seed
        seed = RandomState(1234)

        # split data into train and test sets
        pool = self.X_transformed[train_index]
        oracle = self.y[train_index]
        labels = np.ma.MaskedArray(oracle, mask=True, copy=True)
        X_test = self.X_transformed[test_index]
        y_test = self.y[test_index]
        n_classes = len(np.unique(y_test))
        similarity = rbf_kernel(self.X[train_index], gamma=self.gamma)
        mpba, accuracy, f1 = [], [], []
        training_size = min(1000, len(pool))
        initial_n = 10
        horizon = training_size - initial_n

        # initialise classifier
        classifier = LogisticRegression(multi_class='ovr', penalty='l2', C=1000,
                                        random_state=seed, class_weight='balanced')
        committee = BaggingClassifier(classifier, n_estimators=7, n_jobs=1, max_samples=100,
                                      random_state=seed)

        # select the specified policy
        policy = self._get_policy(self.policy_name, pool, labels, classifier,
                                  committee, seed, similarity, horizon)

        # select 10 initial random examples for labelling
        sample_idx = sample_from_every_class(oracle, initial_n, seed)
        policy.add(sample_idx, oracle[sample_idx])
        y_pred = classifier.predict(X_test)
        mpba.append(mpba_score(y_test, y_pred))
        accuracy.append(accuracy_score(y_test, y_pred))
        f1.append(micro_f1_score(y_test, y_pred, n_classes))

        # start running the policy
        while np.sum(~labels.mask) < training_size:
            # use the policy to select the next instance for labelling
            best_candidates = policy.select()

            # query the oracle and add label
            policy.add(best_candidates, oracle[best_candidates])

            # observe the reward
            y_pred = classifier.predict(X_test)
            mpba.append(mpba_score(y_test, y_pred))
            reward = mpba[-1] - mpba[-2]

            # also compute accuracy and f1 score
            accuracy.append(accuracy_score(y_test, y_pred))
            f1.append(micro_f1_score(y_test, y_pred, n_classes))

            # normalise the reward to [0, 1]
            reward = (reward + 1) / 2
            policy.receive_reward(reward)

        history = policy.history()
        history['mpba'] = np.array(mpba)
        history['accuracy'] = np.array(accuracy)
        history['f1'] = np.array(f1)

        return history

    def _get_policy(self, request, pool, labels, classifier,
                    committee, seed, similarity, horizon):

        similarity = similarity if request.startswith('w-') else None

        arms = [RandomArm(pool, labels, seed)] if self.passive else []
        arms += [
            MarginArm(pool, labels, seed),
            ConfidenceArm(pool, labels, seed),
            EntropyArm(pool, labels, seed),
            QBBMarginArm(pool, labels, committee, 100, seed),
            QBBKLArm(pool, labels, committee, 100, seed)]

        if request == 'passive':
            policy = SingleSuggestion(pool, labels, classifier,
                                      RandomArm(pool, labels, seed))

        elif request in ['margin', 'w-margin']:
            policy = SingleSuggestion(pool, labels, classifier,
                                      MarginArm(pool, labels, seed, similarity))

        elif request in ['confidence', 'w-confidence']:
            policy = SingleSuggestion(pool, labels, classifier,
                                      ConfidenceArm(pool, labels, seed, similarity))

        elif request in ['entropy', 'w-entropy']:
            policy = SingleSuggestion(pool, labels, classifier,
                                      EntropyArm(pool, labels, seed, similarity))

        elif request == 'qbb-margin':
            policy = SingleSuggestion(pool, labels, classifier,
                                      QBBMarginArm(pool, labels, committee, 100, seed))

        elif request == 'qbb-kl':
            policy = SingleSuggestion(pool, labels, classifier,
                                      QBBKLArm(pool, labels, committee, 100, seed))

        elif request == 'thompson':
            policy = ThompsonSampling(pool, labels, classifier, arms, seed)

        elif request == 'ocucb':
            policy = OCUCB(pool, labels, classifier, arms, seed, horizon=horizon)

        elif request == 'klucb':
            policy = KLUCB(pool, labels, classifier, arms, seed)

        elif request == 'exp++':
            policy = EXP3PP(pool, labels, classifier, arms, seed)

        elif request in ['borda', 'geometric']:
            policy = ActiveAggregator(pool, labels, classifier, arms, request, seed)

        elif request == 'schulze':
            policy = ActiveAggregator(pool, labels, classifier, arms, request, seed, 100)

        else:
            raise ValueError('The given policy name {} is not recognised'.format(request))

        return policy
