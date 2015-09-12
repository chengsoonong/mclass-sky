import os
import numpy as np
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from mclearn.active import ActiveLearner, ActiveBandit
from mclearn.heuristics import (random_h, entropy_h, margin_h, qbb_kl_h, qbb_margin_h)
from .datasets import Dataset


class TestActive:
    @classmethod
    def setup_class(cls):
        cls.data = [Dataset('glass'), Dataset('wine')]
        cls.heuristics = [random_h, entropy_h, margin_h, qbb_kl_h, qbb_margin_h]
        cls.classifier = LogisticRegression()
        cls.committee = BaggingClassifier(cls.classifier, n_estimators=2, max_samples=30)


    def _run_active_expt(self, data):
        X_train, X_test, y_train, y_test = train_test_split(data.features, data.target)
        for heuristic in self.heuristics:
            active_learner = ActiveLearner(classifier=self.classifier,
                                           heuristic=heuristic,
                                           initial_n=20,
                                           training_size=23,
                                           sample_size=20,
                                           committee=self.committee,
                                           committee_samples=20,
                                           pool_n = 10,
                                           C=1,
                                           verbose=False)
            active_learner.fit(X_train, y_train, X_test, y_test)
            assert len(active_learner.learning_curve_) == 23 - 20 + 1


    def _run_bandit_expt(self, data):
        X_train, X_test, y_train, y_test = train_test_split(data.features, data.target)
        active_bandit = ActiveBandit(classifier=self.classifier,
                                     heuristics=self.heuristics,
                                     initial_n=20,
                                     training_size=23,
                                     sample_size=20,
                                     committee=self.committee,
                                     committee_samples=20,
                                     pool_n = 10,
                                     C=1,
                                     verbose=False)
        active_bandit.fit(X_train, y_train, X_test, y_test)
        assert len(active_bandit.learning_curve_) == 23 - 20 + 1


    def test_active(self):
        for data in self.data:
            self._run_active_expt(data)
            self._run_bandit_expt(data)
