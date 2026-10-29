"""Checks for research/experiments/block_stability.py."""

import unittest

import numpy as np

from research.experiments.block_stability import radius
from research.experiments.sharpness_floor import boundary


class BlockStabilityTests(unittest.TestCase):
    def test_scalar_case_matches_the_scalar_boundary(self):
        for d, beta in [(1, 0.0), (4, 0.0), (4, 0.9)]:
            b = boundary(d, beta)
            a = np.array([[1.0]])
            self.assertLess(radius(np.zeros((1, 1)), a, 0.98 * b, beta, d), 1)
            self.assertGreater(radius(np.zeros((1, 1)), a, 1.02 * b, beta, d), 1)

    def test_live_curvature_is_not_delayed(self):
        # all curvature live: stable up to eta*lambda < 2 at any delay (no momentum)
        a = np.array([[1.0]])
        self.assertLess(radius(a, np.zeros((1, 1)), 1.9, 0.0, 16), 1)
        self.assertGreater(radius(np.zeros((1, 1)), a, 1.9, 0.0, 16), 1)


if __name__ == "__main__":
    unittest.main()
