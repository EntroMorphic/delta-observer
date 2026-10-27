"""Harness checks for research/experiments/interface_sweep.py."""

import unittest

import torch

from research.experiments.interface_sweep import ARMS, Trainer, straight_through


def batch(trainer, n=64):
    return trainer.inputs[:n], trainer.outputs[:n]


class InterfaceSweepTests(unittest.TestCase):
    def test_every_arm_builds_and_scores(self):
        for arm in ARMS:
            trainer = Trainer(arm, 0, 0)
            self.assertGreaterEqual(trainer.accuracy(), 0.0)
            trainer.step(torch.arange(16))

    def test_per_stage_path_matches_joint_gradient_when_views_are_live(self):
        for arm in ["A_cont_w1", "D_sign_w4", "F_pipeline_w64", "H_pipeline_local_w64"]:
            joint, staged = Trainer(arm, 1, 0), Trainer(arm, 1, 8)
            x, y = batch(joint)
            expected = joint.gradients(x, y)
            got = staged.gradients(x, y)  # history holds only the live snapshot
            for e, g in zip(expected, got):
                torch.testing.assert_close(e, g)

    def test_extrapolation_with_a_constant_history_is_plain_staleness(self):
        stale, extrap = Trainer("A_cont_w1", 2, 4), Trainer("E_extrap_w1", 2, 4)
        x, y = batch(stale)
        for e, g in zip(stale.gradients(x, y), extrap.gradients(x, y)):
            torch.testing.assert_close(e, g)

    def test_stale_stage_sees_others_delayed(self):
        trainer = Trainer("B_cont_w4", 3, 2)
        for k in range(3):
            trainer.step(torch.arange(16 * k, 16 * k + 16))
        x, y = batch(trainer)
        old, live = trainer.view(2), trainer.live()
        grads = trainer.gradients(x, y)
        offset = 0
        for m in range(trainer.net.n):
            views = [live[s] if s == m else old[s] for s in range(trainer.net.n)]
            params = list(live[m].values())
            for e, g in zip(torch.autograd.grad(trainer.full_loss(views, x, y), params), grads[offset:]):
                torch.testing.assert_close(e, g)
            offset += len(params)

    def test_straight_through_is_hard_forward_identity_backward(self):
        raw = torch.tensor([-0.3, 0.2, 0.7, 1.4], requires_grad=True)
        for mode, hard in [("round", [0.0, 0.0, 1.0, 1.0]), ("sign", [-1.0, 1.0, 1.0, 1.0])]:
            out = straight_through(raw, mode)
            self.assertEqual(out.tolist(), hard)
            (grad,) = torch.autograd.grad(out.sum(), raw)
            self.assertEqual(grad.tolist(), [1.0] * 4)

    def test_interface_meaning_is_bounded(self):
        scores = Trainer("A_cont_w1", 0, 0).interface_meaning()
        self.assertEqual(len(scores), 5)
        for s in scores:
            self.assertLessEqual(s, 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
