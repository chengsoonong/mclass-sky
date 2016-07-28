""" Active learning experiment. """

# Author: Alasdair Tran
# License: BSD 3 clause

import os
import numpy as np
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.preprocessing import StandardScaler

from .arms import RandomArm, MarginArm, LeastConfidenceArm, EntropyArm, QBBMarginArm, QBBKLArm
from .performance import mpba_score
from .policies import SingleSuggestion, ThompsonSampling, OCUCB
from .policies import BordaAggregator, GeometricAggregator, SchulzeAggregator


def save_results(dataset, policy, results):
    dir_path = os.path.join('results', dataset, policy)
    os.makedirs(dir_path, exist_ok=True)
    dump_path = os.path.join(dir_path, 'results.pkl')
    joblib.dump(results, dump_path)


class ActiveExperiment:
    """ Simulate an active learning experiment. """
    def __init__(self, X, y, dataset, policy, scale=True):
        seed = RandomState(1234)
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.X = StandardScaler().fit_transform(self.X) if scale else self.X
        self.policy = policy
        self.dataset = dataset

        # estimate the kernel using the 90th percentile heuristic
        random_idx = seed.choice(X.shape[0], 1000)
        distances = pairwise_distances(self.X[random_idx], metric='l1')
        self.gamma = 1 / np.percentile(distances, 90)
        transformer = RBFSampler(gamma=self.gamma, random_state=seed, n_components=100)
        self.X_transformed = transformer.fit_transform(self.X)

        self.kfold = StratifiedShuffleSplit(self.y, n_iter=10, test_size=0.3,
                                            train_size=0.7, random_state=seed)

    def run(self):
        outputs = Parallel(n_jobs=-1)(delayed(self._run_fold)(train_index, test_index)
                                      for train_index, test_index in self.kfold)

        # unpack results
        results = {}
        for key in outputs[0]:
            results[key] = [fold[key] for fold in outputs]
            results[key] = np.asarray(results[key])
        save_results(self.dataset, self.policy, results)

    def _run_fold(self, train_index, test_index):
        # reset the seed
        seed = RandomState(1234)

        # split data into train and test sets
        pool = self.X_transformed[train_index]
        oracle = self.y[train_index]
        labels = np.ma.MaskedArray(oracle, mask=True, copy=True)
        X_test = self.X_transformed[test_index]
        y_test = self.y[test_index]
        similarity = rbf_kernel(self.X[train_index], gamma=self.gamma)
        learning_curve = []

        # initialise classifier
        classifier = LogisticRegression(multi_class='ovr', penalty='l2', C=1000,
                                        random_state=seed, class_weight='balanced')
        committee = BaggingClassifier(classifier, n_estimators=7, n_jobs=1, max_samples=300,
                                      random_state=seed)

        # initialise policy and arms
        arms = [
            RandomArm(pool, labels, seed),
            MarginArm(pool, labels, seed),
            #LeastConfidenceArm(pool, labels, seed),
            EntropyArm(pool, labels, seed),
            QBBMarginArm(pool, labels, committee, 300, seed),
            QBBKLArm(pool, labels, committee, 300, seed)
        ]

        policies = {
            'thompson': ThompsonSampling(pool, labels, classifier, arms, seed, 300),
            'ocucb': OCUCB(pool, labels, classifier, arms, seed, 300),
            'borda': BordaAggregator(pool, labels, classifier, arms, seed, 300),
            'geometric': GeometricAggregator(pool, labels, classifier, arms, seed, 300),
            'schulze': SchulzeAggregator(pool, labels, classifier, arms, seed, 300)
        }

        chosen_policy = policies[self.policy]

        # select 50 initial random examples for labelling
        sample_idx = seed.choice(np.arange(len(pool)), 50, replace=False)
        chosen_policy.add(sample_idx, oracle[sample_idx])
        y_pred = classifier.predict(X_test)
        learning_curve.append(mpba_score(y_test, y_pred))

        # start running the policy
        training_size = min(1000, len(pool))
        while np.sum(~labels.mask) < training_size:
            # use the policy to select the next instance for labelling
            best_candidates = chosen_policy.select()

            # query the oracle and add label
            chosen_policy.add(best_candidates, oracle[best_candidates])

            # observe the reward
            y_pred = classifier.predict(X_test)
            learning_curve.append(mpba_score(y_test, y_pred))
            reward = learning_curve[-1] - learning_curve[-2]
            chosen_policy.receive_reward(reward)

        history = chosen_policy.history()
        history['learning_curve'] = np.array(learning_curve)

        return history
