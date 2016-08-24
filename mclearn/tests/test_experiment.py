import os
import shutil
from .datasets import Dataset
from mclearn.experiment import ActiveExperiment

class TestExperiment:
    @classmethod
    def setup_class(cls):
        cls.data = Dataset('wine')
        cls.policies = ['passive', 'margin', 'w-margin', 'confidence',
                        'w-confidence', 'entropy', 'w-entropy',
                        'qbb-margin', 'qbb-kl', 'thompson', 'ocucb', 'klucb',
                        'exp++', 'borda', 'geometric', 'schulze']

    def test_experiment(self):
        for policy in self.policies:
            expt = ActiveExperiment(self.data.features, self.data.target,
                                    'wine', policy, n_iter=1)
            expt.run_policies()

    @classmethod
    def teardown_class(cls):
        if os.path.exists('results'):
            shutil.rmtree('results')
