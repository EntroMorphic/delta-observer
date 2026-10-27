"""Harness checks for research/experiments/contract_lifecycle.py."""

import unittest

import torch

from research.experiments.contract_lifecycle import OneInterfaceTrainer, branch_loss, messages_and_sensitivity


class ContractLifecycleTests(unittest.TestCase):
    def test_one_interface_is_fresh_when_history_is_live(self):
        a, b = OneInterfaceTrainer("A_cont_w1", 0, 8), OneInterfaceTrainer("A_cont_w1", 0, 8)
        b.interface = 2
        x, y = a.inputs[:64], a.outputs[:64]
        for g, h in zip(a.gradients(x, y), b.gradients(x, y)):
            torch.testing.assert_close(g, h)

    def test_sensitivity_is_the_total_derivative(self):
        trainer = OneInterfaceTrainer("B_cont_w4", 1, 4)
        x, y = trainer.inputs[:32], trainer.outputs[:32]
        messages, sens = messages_and_sensitivity(trainer, x, y)
        # perturb message into stage 3 and rerun downstream: first-order change must match
        from research.experiments.interface_sweep import run_from
        live = [{k: p.detach() for k, p in v.items()} for v in trainer.live()]
        eps = 1e-3 * torch.randn_like(messages[2])
        base = run_from(trainer.net, live, x, y, 3, messages[2]) * len(x) * y.shape[1]
        moved = run_from(trainer.net, live, x, y, 3, messages[2] + eps) * len(x) * y.shape[1]
        self.assertAlmostEqual(float(moved - base), float((sens[2] * eps).sum()), delta=1e-4)

    def test_branch_leaves_trainer_untouched(self):
        trainer = OneInterfaceTrainer("A_cont_w1", 2, 4)
        before = [p.clone() for p in trainer.net.parameters()]
        branch_loss(trainer, 1, 0)
        for p, q in zip(before, trainer.net.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
