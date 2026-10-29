"""Checks for research/optimizers/hybrid_muon.py and the hybrid harness."""

import unittest

import torch

from research.experiments.hybrid_staleness import HYBRIDS, build_hybrid
from research.optimizers.hybrid_muon import AdaMuonStyleMatrices, PreconditionedMuonMatrices


class HybridMuonTests(unittest.TestCase):
    def test_both_hybrids_move_every_parameter(self):
        for name in HYBRIDS:
            trainer = build_hybrid("B_cont_w4", 0, "fresh", 0, name, 0.01)
            before = [p.clone() for p in trainer.net.parameters()]
            for k in range(3):
                trainer.step(torch.arange(32 * k, 32 * k + 32))
            for p, q in zip(before, trainer.net.parameters()):
                self.assertFalse(torch.equal(p, q), name)

    def test_adamuon_style_matches_muon_rms(self):
        p = torch.nn.Parameter(torch.randn(8, 4))
        opt = AdaMuonStyleMatrices([p], lr=1.0, momentum=0.0)
        p.grad = torch.randn(8, 4)
        from research.optimizers.muon import zeroth_power_newton_schulz
        expected = zeroth_power_newton_schulz(p.grad).norm() * 2 ** 0.5
        before = p.detach().clone()
        opt.step()
        self.assertAlmostEqual(float((before - p.detach()).norm()), float(expected), places=4)

    def test_preconditioning_happens_before_orthogonalization(self):
        p = torch.nn.Parameter(torch.zeros(4, 4))
        opt = PreconditionedMuonMatrices([p], lr=1.0, momentum=0.0)
        p.grad = torch.diag(torch.tensor([1.0, 10.0, 100.0, 1000.0]))
        opt.step()
        # after per-coordinate normalization the diagonal is uniform, so the
        # orthogonalized step is (close to) a uniform diagonal
        d = -p.detach().diagonal()
        self.assertLess(float(d.max() - d.min()), 0.1 * float(d.mean()))


if __name__ == "__main__":
    unittest.main()
