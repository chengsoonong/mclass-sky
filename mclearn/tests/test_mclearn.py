from mclearn import *

class TestMCLearn:
    def test_import_mclearn(self):
        packages = ['active', 'classifier', 'heuristics', 'performance',
                    'photometry', 'preprocessing', 'tools', 'viz']
        for package in packages:
            assert package in globals()