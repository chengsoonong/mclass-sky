import os
import numpy as np
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from mclearn.active import ActiveLearner, ActiveBandit
from mclearn.heuristics import random_h, entropy_h, margin_h, qbb_kl_h, qbb_margin_h


class TestActive:
    @classmethod
    def setup_class(cls):
        cls.glass = read_csv('mclearn/tests/data/glass.csv')
        cls.feature_cols = ['ri', 'na', 'mg', 'ai', 'si', 'k', 'ca', 'ba', 'fe']
        cls.target_col = 'type'
        cls.heuristics = [random_h, entropy_h, margin_h, qbb_kl_h, qbb_margin_h]
        cls.initial_n = 20
        cls.training_size = 23

        cls.X = cls.glass[cls.feature_cols]
        cls.y = cls.glass[cls.target_col]
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y)

        cls.logistic_classifier = LogisticRegression()
        cls.committee = BaggingClassifier(cls.logistic_classifier, n_estimators=2, max_samples=30)


    def test_active_learner(self):
        for heuristic in self.heuristics:
            active_learner = ActiveLearner(classifier=self.logistic_classifier,
                                           heuristics=heuristic,
                                           initial_n=self.initial_n,
                                           training_size=self.training_size,
                                           sample_size=len(self.y),
                                           committee=self.committee,
                                           verbose=False)
            active_learner.fit(self.X_train, self.y_train, self.X_test, self.y_test)
            assert len(active_learner.learning_curve_) == self.training_size - self.initial_n + 1


    def test_active_bandit(self):
        active_bandit = ActiveBandit(classifier=self.logistic_classifier,
                                     heuristics=self.heuristics,
                                     initial_n=self.initial_n,
                                     training_size=self.training_size,
                                     sample_size=len(self.y),
                                     committee=self.committee,
                                     verbose=False)
        active_bandit.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        assert len(active_bandit.learning_curve_) == self.training_size - self.initial_n + 1