import unittest

import numpy as np

from src.Features.MAD import mean_absolute_deviation

class TestFeatures(unittest.TestCase):
    def test_mad(self):
        data = np.array([5, 12, 1, 0, 4, 22, 15, 3, 9])
        self.assertAlmostEqual(mean_absolute_deviation(data), 5.876543209876543)


if __name__ == '__main__':
    unittest.main()
