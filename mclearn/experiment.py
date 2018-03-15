""" Active learning experiment. """

# Author: Alasdair Tran
# License: BSD 3 clause

import os
import numpy as np
from time import time
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from mclearn.arms import RandomArm, MarginArm, ConfidenceArm, EntropyArm, QBBMarginArm, QBBKLArm
from mclearn.performance import mpba_score, micro_f1_score
from mclearn.policies import (
    BaselineCombiner,
    SingleSuggestion,
    ThompsonSampling,
    OCUCB,
    KLUCB,
    EXP3PP,
    ActiveAggregator,
    COMB,
)


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
        if (len(labels) == 0 and idx not in samples) or y[idx] in labels:
            samples.append(idx)
            labels = np.delete(labels, np.argwhere(labels == y[idx]))
    return samples


class ActiveExperiment:
    """ Simulate an active learning experiment. """
    def __init__(self, X, y, dataset, policy_name, scale=True, n_splits=10,
                 passive=True, n_jobs=-1, overwrite=False,
                 gamma_percentile=90, ts_sigma=0.02, ts_tau=0.02, ts_mu=0.5,
                 save_name=None, candidate_pool_size=None):
        seed = RandomState(1234)
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y)
        self.X = StandardScaler().fit_transform(self.X) if scale else self.X
        self.policy_name = policy_name
        self.dataset = dataset
        self.passive = passive
        self.n_jobs = n_jobs
        self.overwrite = overwrite
        self.ts_sigma = ts_sigma
        self.ts_tau = ts_tau
        self.ts_mu = ts_mu
        self.save_name = save_name
        self.candidate_pool_size=candidate_pool_size

        # estimate the kernel using the 90th percentile heuristic
        random_idx = seed.choice(X.shape[0], 1000)
        distances = pairwise_distances(self.X[random_idx], metric='l1')
        self.gamma = 1 / np.percentile(distances, 90)
        self.similarity_gamma = 1 / np.percentile(distances, gamma_percentile)
        transformer = RBFSampler(gamma=self.gamma, random_state=seed, n_components=100)
        self.X_transformed = transformer.fit_transform(self.X)

        n_samples = self.X.shape[0]
        train_size = min(10000, int(0.7 * n_samples))
        test_size = min(20000, n_samples - train_size)
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, train_size=train_size, test_size=test_size, random_state=seed)
        self.kfold = list(splitter.split(self.X, self.y))

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)

        if policy_name == 'COMB':
            assert len(self.label_encoder.classes_) == 2, 'COMB only works with binary classification.'

    def run_policies(self):
        if not self.save_name:
            self.save_name = self.policy_name if self.passive else self.policy_name + '-wop'

        if not self.overwrite and os.path.exists(os.path.join('results', self.dataset, self.save_name)):
            return

        start_time = time()
        outputs = Parallel(n_jobs=self.n_jobs)(delayed(self._run_fold)(train_index, test_index)
                                      for train_index, test_index in self.kfold)
        end_time = time()

        # unpack results
        results = {}
        for key in outputs[0]:
            results[key] = [fold[key] for fold in outputs]
            results[key] = np.asarray(results[key])
            results['time'] = end_time - start_time

        save_results(self.dataset, self.save_name, results)

    def run_asymptote(self):
        if not self.overwrite and os.path.exists(os.path.join('results', self.dataset, 'asymptote')):
            return

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
            grid = self.grid_search(seed, X_train, y_train)
            classifier = LogisticRegression(multi_class='ovr', penalty='l2', C=grid.best_params_['C'],
                                            random_state=seed, class_weight='balanced')
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            results['asymptote_mpba'].append(mpba_score(y_test, y_pred))
            results['asymptote_accuracy'].append(accuracy_score(y_test, y_pred))
            results['asymptote_f1'].append(micro_f1_score(y_test, y_pred, n_classes))

        for key in results:
            results[key] = np.array(results[key])

        save_results(self.dataset, 'asymptote', results)

    def grid_search(self, seed, pool, oracle):
        # Conduct a grid search to find the best C
        classifier = LogisticRegression(multi_class='ovr', penalty='l2', C=1000,
                                        random_state=seed, class_weight='balanced')
        grid_test_size = int(min(100, 0.3 * len(pool)))
        grid_train_size = int(min(100, 0.7 * len(pool)))
        cv = StratifiedShuffleSplit(n_splits=5, train_size=grid_train_size,
                                    test_size=grid_test_size, random_state=17)
        C_range = np.logspace(-6, 6, 13)
        param_grid = dict(C=C_range)
        grid = GridSearchCV(classifier, param_grid)
        grid.fit(pool, oracle)
        return grid

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
        similarity = rbf_kernel(self.X[train_index], gamma=self.similarity_gamma)
        mpba, accuracy, f1 = [], [], []
        training_size = min(1000, len(pool))
        initial_n = 10
        horizon = training_size - initial_n

        grid = self.grid_search(seed, pool, oracle)

        # initialise classifier
        classifier = LogisticRegression(multi_class='ovr', penalty='l2', C=grid.best_params_['C'],
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

        if isinstance(policy, COMB):
            entropy = [comb_entropy(policy, pool, self.label_encoder)]

        # start running the policy
        while np.sum(~labels.mask) < training_size:
            # use the policy to select the next instance for labelling
            best_candidates = policy.select()

            # query the oracle and add label
            policy.add(best_candidates, oracle[best_candidates])

            # observe the reward
            if isinstance(policy, COMB):
                entropy.append(comb_entropy(policy, pool, self.label_encoder))

                # Calculate reward utility
                reward = ((np.exp(entropy[-1]) - np.exp(entropy[-1])) - (1 - np.exp(1))) / (2 * np.exp(1) - 2)
            else:
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
        if isinstance(policy, COMB):
            history['entropy'] = np.array(entropy)

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
                                      RandomArm(pool, labels, seed),
                                      n_candidates=self.n_candidates)

        elif request in ['margin', 'w-margin']:
            policy = SingleSuggestion(pool, labels, classifier,
                                      MarginArm(pool, labels, seed, similarity),
                                      n_candidates=self.n_candidates)

        elif request in ['confidence', 'w-confidence']:
            policy = SingleSuggestion(pool, labels, classifier,
                                      ConfidenceArm(pool, labels, seed, similarity),
                                      n_candidates=self.n_candidates)

        elif request in ['entropy', 'w-entropy']:
            policy = SingleSuggestion(pool, labels, classifier,
                                      EntropyArm(pool, labels, seed, similarity),
                                      n_candidates=self.n_candidates)

        elif request == 'qbb-margin':
            policy = SingleSuggestion(pool, labels, classifier,
                                      QBBMarginArm(pool, labels, committee, 100, seed),
                                      n_candidates=self.n_candidates)

        elif request == 'qbb-kl':
            policy = SingleSuggestion(pool, labels, classifier,
                                      QBBKLArm(pool, labels, committee, 100, seed),
                                      n_candidates=self.n_candidates)

        elif request == 'thompson':
            policy = ThompsonSampling(pool, labels, classifier, arms, seed,
                                      sigma=self.ts_sigma, tau=self.ts_tau,
                                      n_candidates=self.n_candidates)

        elif request == 'baseline':
            policy = BaselineCombiner(pool, labels, classifier, arms, seed,
                                      n_candidates=self.n_candidates)

        elif request == 'ocucb':
            policy = OCUCB(pool, labels, classifier, arms, seed, horizon=horizon,
                           n_candidates=self.n_candidates)

        elif request == 'klucb':
            policy = KLUCB(pool, labels, classifier, arms, seed,
                           n_candidates=self.n_candidates)

        elif request == 'exp++':
            policy = EXP3PP(pool, labels, classifier, arms, seed,
                            n_candidates=self.n_candidates)

        elif request in ['borda', 'geometric']:
            policy = ActiveAggregator(pool, labels, classifier, arms, request, seed,
                                      n_candidates=self.n_candidates)

        elif request == 'schulze':
            policy = ActiveAggregator(pool, labels, classifier, arms, request, seed, 100)

        elif request == 'comb':
            policy = COMB(pool, labels, classifier, arms, seed,
                          n_candidates=self.n_candidates)

        else:
            raise ValueError('The given policy name {} is not recognised'.format(request))

        return policy

def comb_entropy(policy, pool, label_encoder):
    unlabeled_idx = policy.labels.mask
    pred = policy.classifier.predict(pool[unlabeled_idx])
    neg_class, pos_class = label_encoder.classes_
    n_neg = np.sum(pred == neg_class)
    n_pos = np.sum(pred == pos_class)
    assert n_neg + n_pos == len(pred), 'Mismatch between negative and positive count'

    # Calculate binary entropy
    p = n_pos / (n_neg + n_pos)
    h1 = -p * np.log2(p) if p > 0 else 0
    h2 = - (1 - p) * np.log2(1 - p) if p < 1 else 0
    entropy = h1 + h2

    return entropy
