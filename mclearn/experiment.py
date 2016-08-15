""" Active learning experiment. """

# Author: Alasdair Tran
# License: BSD 3 clause

import os
import numpy as np
from time import time
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.cross_validation import StratifiedShuffleSplit
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


class ActiveExperiment:
    """ Simulate an active learning experiment. """
    def __init__(self, X, y, dataset, policy_name, scale=True, n_iter=10):
        seed = RandomState(1234)
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        self.X = StandardScaler().fit_transform(self.X) if scale else self.X
        self.policy_name = policy_name
        self.dataset = dataset

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

    def run(self):
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
        save_results(self.dataset, self.policy_name, results)

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
        initial_n = 50
        horizon = training_size - initial_n

        # initialise classifier
        classifier = LogisticRegression(multi_class='ovr', penalty='l2', C=1000,
                                        random_state=seed, class_weight='balanced')
        committee = BaggingClassifier(classifier, n_estimators=7, n_jobs=1, max_samples=100,
                                      random_state=seed)

        # select the specified policy
        policy = self._get_policy(self.policy_name, pool, labels, classifier,
                                  committee, seed, similarity, horizon)

        # select 50 initial random examples for labelling
        sample_idx = seed.choice(np.arange(len(pool)), initial_n, replace=False)
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

        similarity = similarity if request.startswith('weighted') else None

        arms = [
            RandomArm(pool, labels, seed),
            MarginArm(pool, labels, seed),
            ConfidenceArm(pool, labels, seed),
            EntropyArm(pool, labels, seed),
            QBBMarginArm(pool, labels, committee, 100, seed),
            QBBKLArm(pool, labels, committee, 100, seed)
        ]

        if request == 'passive':
            policy = SingleSuggestion(pool, labels, classifier,
                                      RandomArm(pool, labels, seed))

        elif request in ['margin', 'weighted-margin']:
            policy = SingleSuggestion(pool, labels, classifier,
                                      MarginArm(pool, labels, seed, similarity))

        elif request in ['confidence', 'weighted-confidence']:
            policy = SingleSuggestion(pool, labels, classifier,
                                      ConfidenceArm(pool, labels, seed, similarity))

        elif request in ['entropy', 'weighted-entropy']:
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
