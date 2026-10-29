"""Checks for research/experiments/gradient_floor.py."""

import unittest

import torch

from research.experiments.gradient_floor import descent_ratios, make
from research.experiments.sharpness_floor import GRID


class GradientFloorTests(unittest.TestCase):
    def test_ratio_is_one_without_history_and_training_stays_fresh(self):
        trainer = make("sgd", 0.1, 0)
        x, y = trainer.inputs[:256], trainer.outputs[:256]
        r = descent_ratios(trainer, x, y)  # history holds only the current snapshot
        for d in GRID:
            self.assertAlmostEqual(r[d], 1.0, places=5)
        self.assertEqual(trainer.delay, 0)

    def test_sgd0_has_no_momentum_and_keeps_long_history(self):
        trainer = make("sgd0", 1.0, 0)
        self.assertEqual(trainer.opt.param_groups[0]["momentum"], 0.0)
        for k in range(80):
            trainer.step(torch.arange(32 * (k % 100), 32 * (k % 100) + 32))
        self.assertGreaterEqual(len(trainer.history), max(GRID) + 1)


if __name__ == "__main__":
    unittest.main()
