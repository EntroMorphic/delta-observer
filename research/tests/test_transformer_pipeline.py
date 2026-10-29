"""Checks for research/experiments/transformer_pipeline.py."""

import unittest

import torch

from research.experiments.transformer_pipeline import STAGES, PipelineRun


class TransformerPipelineTests(unittest.TestCase):
    def test_stage_delays_and_assignment(self):
        run = PipelineRun("stale", 12, 1e-3, 0)
        self.assertEqual(run.tau, [12, 8, 4, 0])
        self.assertEqual(sorted(set(run.stage)), list(range(STAGES)))
        self.assertEqual(run.stage[run.names.index("tok.weight")], 0)
        self.assertEqual(run.stage[run.names.index("head.weight")], STAGES - 1)

    def test_stale_gradient_matches_fresh_before_any_history(self):
        fresh, stale = PipelineRun("fresh", 0, 1e-3, 0), PipelineRun("stale", 12, 1e-3, 0)
        x, y = fresh.batch(fresh.train_ids, torch.Generator().manual_seed(1))
        for a, b in zip(fresh.gradients(x, y), stale.gradients(x, y)):
            torch.testing.assert_close(a, b)

    def test_snapshot_is_weights_from_tau_steps_ago(self):
        run = PipelineRun("stale", 6, 1e-3, 0)
        seen = []
        for _ in range(8):
            seen.append([p.detach().clone() for p in run.params])
            run.step()
        for tau in (1, 4, 6):
            weights, update = run.snapshot(tau)
            torch.testing.assert_close(weights[0], seen[-tau][0])
            nxt = seen[-tau + 1][0] if tau > 1 else run.params[0].detach()
            torch.testing.assert_close(weights[0] + update[0], nxt)

    def test_permuted_twin_keeps_error_norm_per_tensor(self):
        run = PipelineRun("permuted", 6, 1e-3, 0)
        for _ in range(8):
            run.step()
        x, y = run.batch(run.train_ids, torch.Generator().manual_seed(2))
        twin = run.gradients(x, y)
        run.method = "stale"
        stale = run.gradients(x, y)
        fresh = torch.autograd.grad(run.loss_at(run.params, x, y), run.params)
        for t, s, f in zip(twin, stale, fresh):
            torch.testing.assert_close((t - f).norm(), (s - f).norm(), rtol=1e-4, atol=1e-6)

    def test_pipemare_scales_early_stage_learning_rates(self):
        run = PipelineRun("pipemare_lr", 12, 1e-3, 0)
        run.set_lr()
        lrs = {k: g["lr"] for g, k in run.groups}
        self.assertAlmostEqual(lrs[0] * 12, lrs[3], places=10)


if __name__ == "__main__":
    unittest.main()
