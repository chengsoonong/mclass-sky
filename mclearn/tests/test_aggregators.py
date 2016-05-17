from mclearn.aggregators import schulze_method


class TestAggregators:
    @classmethod
    def setup_class(cls):
        cls.voters = \
            [['A', 'C', 'B', 'E', 'D']] * 5 + \
            [['A', 'D', 'E', 'C', 'B']] * 5 + \
            [['B', 'E', 'D', 'A', 'C']] * 8 + \
            [['C', 'A', 'B', 'E', 'D']] * 3 + \
            [['C', 'A', 'E', 'B', 'D']] * 7 + \
            [['C', 'B', 'A', 'D', 'E']] * 2 + \
            [['D', 'C', 'E', 'B', 'A']] * 7 + \
            [['E', 'B', 'A', 'D', 'C']] * 8


    def test_schulze_method(self):
        # verify example on http://wiki.electorama.com/wiki/Schulze_method
        assert schulze_method(self.voters, 5) == ['E', 'A', 'C', 'B', 'D']

