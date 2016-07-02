import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score
from time import time
from mclearn.heuristics import random_h, entropy_h, margin_h, qbb_kl_h, qbb_margin_h, random_h
from mclearn.aggregators import borda_count, geometric_mean, schulze_method
from mclearn.performance import mpba_score, micro_f1_score
from mclearn.active import ActiveBandit, ActiveLearner, ActiveAggregator
from mclearn.tools import log


def run_active_expt(X, y, dataset, kind, kfold=None, classifier=None, random_state=1234,
    committee=None, heuristics=None, accuracy_fns=None, initial_n=50, training_size=1000,
    sample_size=300, verbose=False, committee_samples=300, degree=1, rbf_kernel=True,
    scale=True, overwrite=False, test_size=0.3, train_size=0.7, n_classes=None, n_jobs=-1):
    """ Run Bandit experiment. """

    if type(random_state) is RandomState:
        seed = random_state
    else:
        seed = RandomState(random_state)

    if not overwrite and results_exist(kind, dataset):
        log(' skipped')
        return

    # need to be numpy arrays as some indexing syntax works differently with DataFrames
    X, y = np.asarray(X), np.asarray(y)
    X = X.astype(np.float64)

    # fall back to defaults if not specified by user
    if not classifier:
        classifier = LogisticRegression(multi_class='ovr', penalty='l2', C=1000,
                                        random_state=seed, class_weight='balanced')
    if not committee:
        committee = BaggingClassifier(classifier, n_estimators=7, n_jobs=1, max_samples=300,
                                      random_state=seed)
    if not heuristics:
        heuristics = [random_h, entropy_h, margin_h, qbb_margin_h, qbb_kl_h]
    if not kfold:
        kfold = StratifiedShuffleSplit(y, n_iter=10, test_size=test_size,
                                       train_size=train_size, random_state=seed)
    if scale:
        X = StandardScaler().fit_transform(X)
    if degree > 1:
        transformer = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
        X = transformer.fit_transform(X)
    elif rbf_kernel == True:
        # estimate the kernel using the 90th percentile heuristic
        random_idx = seed.choice(X.shape[0], 1000)
        distances = pairwise_distances(X[random_idx], metric='l1')
        gamma = 1 / np.percentile(distances, 90)
        transformer = RBFSampler(gamma=gamma, random_state=seed, n_components=100)
        X = transformer.fit_transform(X)
    if not accuracy_fns:
        accuracy_fns = {
            'mpba': mpba_score,
            # too expensive to calculate f1
            #'f1': lambda y_true, y_pred: micro_f1_score(y_true, y_pred, n_classes=n_classes),
            'accuracy': accuracy_score
        }

    n_classes = len(np.unique(y))
    start_time = time()

    outputs = Parallel(n_jobs=n_jobs)(delayed(_run_active_fold)(X, y, kind, classifier,
                                                                random_state, committee,
                                                                heuristics, accuracy_fns,
                                                                initial_n, training_size,
                                                                sample_size, verbose,
                                                                committee_samples, seed,
                                                                train_index, test_index)
                                      for (train_index, test_index) in kfold)

    # unpack outputs
    results = {}
    for key in outputs[0]:
        results[key] = [fold[key] for fold in outputs]
        results[key] = np.asarray(results[key])
    save_results(kind, dataset, results)

    end_time = time()
    log(' {0:.1f} mins'.format((end_time - start_time) / 60 ))


def _run_active_fold(X, y, kind, classifier, random_state, committee, heuristics,
    accuracy_fns, initial_n, training_size, sample_size, verbose, committee_samples,
    seed, train_index, test_index):
    """ Helper function. """

    output = {}
    bandit_algos = ('thompson', 'exp3pp', 'kl-ucb', 'oc-ucb')
    rank_algos = {'borda': borda_count, 'schulze': schulze_method, 'geometric': geometric_mean}

    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    curr_training_size = min(training_size, X_train.shape[0])

    if kind in bandit_algos:
        learner = ActiveBandit(classifier=classifier,
                               heuristics=heuristics,
                               accuracy_fn=accuracy_fns,
                               initial_n=initial_n,
                               training_size=curr_training_size,
                               sample_size=sample_size,
                               committee=committee,
                               committee_samples=committee_samples,
                               verbose=verbose,
                               random_state=seed)
    elif kind == 'passive':
        learner = ActiveLearner(classifier=classifier,
                                heuristic=random_h,
                                accuracy_fn=accuracy_fns,
                                initial_n=initial_n,
                                training_size=curr_training_size,
                                sample_size=sample_size,
                                verbose=verbose,
                                random_state=seed)
    elif kind in rank_algos:
        learner = ActiveAggregator(classifier=classifier,
                                   heuristics=heuristics,
                                   accuracy_fn=accuracy_fns,
                                   initial_n=initial_n,
                                   training_size=curr_training_size,
                                   sample_size=sample_size,
                                   committee=committee,
                                   committee_samples=committee_samples,
                                   verbose=verbose,
                                   aggregator=rank_algos[kind],
                                   random_state=seed)

    learner.fit(X_train, y_train, X_test, y_test)
    for accuracy in accuracy_fns:
        output[accuracy] = learner.learning_curve_[accuracy]

    if kind in bandit_algos:
        output['heuristic'] = learner.heuristic_selection
        output['mu'] = learner.all_prior_mus
        output['signma'] = learner.all_prior_sigmas

    return output


def results_exist(kind, dataset):
    path = os.path.join('results', kind, dataset)
    return os.path.isdir(path)

def save_results(kind, dataset, results):
    dir_path = os.path.join('results', kind, dataset)
    os.makedirs(dir_path, exist_ok=True)
    dump_path = os.path.join(dir_path, 'results.pkl')
    joblib.dump(results, dump_path)

def load_results(kind, dataset, measure='mpba', mean=True):
    dump_path = os.path.join('results', kind, dataset, 'results.pkl')
    results = joblib.load(dump_path)[measure]
    if mean:
        results = results.mean(axis=0)

    return results
