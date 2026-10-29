"""Harness checks for research/experiments/staleness_update_probe.py."""

import unittest

import torch

from research.experiments.interface_confirm import build
from research.experiments.staleness_update_probe import loss_after, step_delta


class UpdateProbeTests(unittest.TestCase):
    def test_step_delta_matches_a_real_step_and_leaves_state_alone(self):
        for optimizer, lr in [("adam", 1 / 300), ("sgd", 0.3), ("muon", 0.03)]:
            trainer = build("adder", "B_cont_w4", 0, "fresh", 0, optimizer, lr)
            for k in range(6):  # past warmup's zero first step
                trainer.step(torch.arange(32 * k, 32 * k + 32))
            x, y = trainer.inputs[:64], trainer.outputs[:64]
            grads = trainer.gradients(x, y)
            before = [p.detach().clone() for p in trainer.net.parameters()]
            predicted = step_delta(trainer, grads)
            for p, b in zip(trainer.net.parameters(), before):
                torch.testing.assert_close(p.detach(), b, rtol=0, atol=0)
            for p, g in zip(trainer.net.parameters(), grads):
                p.grad = g
            trainer.opt.step()
            for p, b, d in zip(trainer.net.parameters(), before, predicted):
                torch.testing.assert_close(p.detach() - b, d)

    def test_loss_after_zero_step_is_current_loss(self):
        trainer = build("adder", "A_cont_w1", 0, "fresh", 0, "adam", 1 / 300)
        zero = [torch.zeros_like(p) for p in trainer.net.parameters()]
        with torch.no_grad():
            current = float(trainer.full_loss(trainer.live(), trainer.inputs, trainer.outputs))
        self.assertAlmostEqual(loss_after(trainer, zero), current, places=6)


if __name__ == "__main__":
    unittest.main()
