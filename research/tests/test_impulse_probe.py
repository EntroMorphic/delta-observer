"""Checks for research/experiments/impulse_probe.py."""

import unittest

import torch

from research.experiments.impulse_probe import CHECKPOINTS, branch, make
from research.experiments.interface_sweep import Trainer


class ImpulseProbeTests(unittest.TestCase):
    def test_identical_branches_have_zero_gap_and_leave_trainer_alone(self):
        for optimizer, lr in [("adam", 1 / 300), ("muon", 0.03), ("muon_pre", 0.03)]:
            trainer = make(optimizer, lr, 4, 0)
            trainer.delay = 0
            for k in range(8):
                trainer.step(torch.arange(32 * k, 32 * k + 32))
            before = [p.detach().clone() for p in trainer.net.parameters()]
            batches = [torch.arange(64 * k, 64 * k + 64) for k in range(max(CHECKPOINTS))]
            g = Trainer.gradients(trainer, trainer.inputs[batches[0]], trainer.outputs[batches[0]])
            a, b = branch(trainer, g, batches), branch(trainer, g, batches)
            self.assertEqual(a, b)
            for p, q in zip(before, trainer.net.parameters()):
                torch.testing.assert_close(p, q.detach(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
