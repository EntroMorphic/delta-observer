"""Checks for research/experiments/stress_floor.py."""

import unittest

import torch

from research.experiments.gradient_floor import make
from research.experiments.stress_floor import branch_loss


class StressFloorTests(unittest.TestCase):
    def test_branches_are_reproducible_and_leave_the_trainer_alone(self):
        trainer = make("sgd", 0.1, 0)
        for k in range(20):
            trainer.step(torch.arange(32 * k, 32 * k + 32))
        before = [p.detach().clone() for p in trainer.net.parameters()]
        batches = [torch.arange(64 * k, 64 * k + 64) for k in range(10)]
        self.assertEqual(branch_loss(trainer, 0, batches), branch_loss(trainer, 0, batches))
        self.assertNotEqual(branch_loss(trainer, 0, batches), branch_loss(trainer, 8, batches))
        for p, q in zip(before, trainer.net.parameters()):
            torch.testing.assert_close(p, q.detach(), rtol=0, atol=0)
        self.assertEqual(trainer.delay, 0)


if __name__ == "__main__":
    unittest.main()
