import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from time import time
from mclearn.heuristics import random_h, entropy_h, margin_h, qbb_kl_h, qbb_margin_h, random_h
from mclearn.performance import mpba_score, micro_f1_score
from mclearn.active import ActiveBandit, ActiveLearner
from mclearn.tools import log


def run_thompson_bandit_expt(X, y, results, dataset, kfold=None, classifier=None,
    committee=None, heuristics=None, accuracy_fns=None, initial_n=50, training_size=1000,
    sample_size=300, verbose=False, committee_samples=300, degree=2, scale=True, kind='bandit'):
    """ Run Bandit experiment. """
    
    X, y = np.asarray(X), np.asarray(y)
    if not classifier:
        classifier = LogisticRegression(multi_class='ovr', penalty='l1', C=1,
                                        random_state=1234, class_weight='balanced')
    if not committee:
        committee = BaggingClassifier(classifier, n_estimators=12,
                                      n_jobs=-1, max_samples=300)
    if not heuristics:
        heuristics = [random_h, entropy_h, margin_h, qbb_margin_h, qbb_kl_h]
    if not kfold:
        kfold = StratifiedShuffleSplit(y, n_iter=10, test_size=0.3, random_state=2345)
    if scale:
        X = StandardScaler().fit_transform(X)
    if degree > 1:
        transformer = PolynomialFeatures(degree=degree, interaction_only=False,
                                         include_bias=True)
        X = transformer.fit_transform(X)
    if not accuracy_fns:
        accuracy_fns = {
            'mpba': mpba_score,
            'f1': micro_f1_score,
            'accuracy': lambda clf, X, y: accuracy_score(y, clf.predict(X))
        }
        
    n_classes = len(np.unique(y))
    start_time = time()
    for i, (train_index, test_index) in enumerate(kfold):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        curr_training_size = min(training_size, X_train.shape[0])

        
        if kind == 'bandit':
            active_bandit = ActiveBandit(classifier=classifier,
                                         heuristics=heuristics,
                                         accuracy_fn=accuracy_fns,
                                         initial_n=initial_n,
                                         training_size=curr_training_size,
                                         sample_size=sample_size,
                                         committee=committee,
                                         committee_samples=committee_samples,
                                         verbose=verbose)

            active_bandit.fit(X_train, y_train, X_test, y_test)

            result = [
                active_bandit.learning_curve_['mpba'],
                active_bandit.learning_curve_['f1'],
                active_bandit.learning_curve_['accuracy'],
                active_bandit.candidate_selections,
                {
                    'heuristics': active_bandit.heuristic_selection,
                    'mu': active_bandit.all_prior_mus,
                    'sigma': active_bandit.all_prior_sigmas
                }
            ]
            
            results.loc['bandit', 'thompson', 'mpba',
                        'logistic_ovr', dataset, n_classes, i] = result
        elif kind == 'random':
            learner = ActiveLearner(classifier=classifier,
                                    heuristic=random_h,
                                    accuracy_fn=accuracy_fns,
                                    initial_n=initial_n,
                                    training_size=curr_training_size,
                                    sample_size=sample_size,
                                    verbose=verbose)
            result = [
                learner.learning_curve_['mpba'],
                learner.learning_curve_['f1'],
                learner.learning_curve_['accuracy'],
                learner.candidate_selections,
                {}
            ]
            learner.fit(X_train, y_train, X_test, y_test)
            
            results.loc['random', 'random', 'mpba',
                        'logistic_ovr', dataset, n_classes, i] = result
        
        log('.', end='')
    end_time = time()
    log("{0:.1f} mins".format((end_time - start_time) / 60 ))