"""Checks for research/experiments/transformer_pipeline_probe.py."""

import unittest

import torch

import research.experiments.transformer_pipeline as tp
from research.experiments.transformer_pipeline_probe import curvature, flat, probe


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.run = tp.PipelineRun("stale", 6, 1e-3, 0)
        for _ in range(10):
            self.run.step()
        self.x, self.y = self.run.batch(self.run.train_ids, torch.Generator().manual_seed(1))

    def test_curvature_matches_finite_difference(self):
        v = torch.randn(sum(p.numel() for p in self.run.params), generator=torch.Generator().manual_seed(2))
        v /= v.norm()

        def grad_at(eps):
            ps, o = [], 0
            for p in self.run.params:
                ps.append((p.detach() + eps * v[o:o + p.numel()].reshape(p.shape)).requires_grad_(True))
                o += p.numel()
            return flat(torch.autograd.grad(self.run.loss_at(ps, self.x, self.y), ps))
        fd = float((grad_at(1e-3) - grad_at(-1e-3)) @ v / 2e-3)
        self.assertAlmostEqual(curvature(self.run, self.x, self.y, v), fd, delta=0.02 * abs(fd) + 1e-6)

    def test_twin_has_the_error_norm(self):
        out = probe(self.run, self.x, self.y)
        self.assertGreater(out["error_norm"], 0)
        self.assertLess(abs(out["cos_twin"]), 0.2)


if __name__ == "__main__":
    unittest.main()
