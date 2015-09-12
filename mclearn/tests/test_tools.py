import os
from mclearn.tools import results_exist
from .datasets import Dataset


class TestTools:
    @classmethod
    def setup_class(cls):
        cls.glass = Dataset('glass')
        cls.wine = Dataset('wine')
        cls.file_paths = [cls.glass.path, cls.wine.path]


    def test_results_exist(self):
        assert results_exist(self.file_paths)

