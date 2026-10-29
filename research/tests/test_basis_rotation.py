"""Checks for research/optimizers/basis_rotation.py and its harness."""

import unittest

import torch

from research.experiments.basis_rotation_bound import make
from research.optimizers.basis_rotation import BasisRotationMatrices


class BasisRotationTests(unittest.TestCase):
    def test_bases_stay_orthonormal(self):
        p = torch.nn.Parameter(torch.randn(6, 4))
        opt = BasisRotationMatrices([p], lr=1e-2, freq=2)
        for _ in range(7):
            p.grad = torch.randn(6, 4)
            opt.step()
        st = opt.state[p]
        torch.testing.assert_close(st["U"].T @ st["U"], torch.eye(6), atol=1e-5, rtol=0)
        torch.testing.assert_close(st["V"].T @ st["V"], torch.eye(4), atol=1e-5, rtol=0)

    def test_identity_basis_reproduces_adam(self):
        torch.manual_seed(0)
        p, q = torch.nn.Parameter(torch.randn(5, 3)), None
        q = torch.nn.Parameter(p.detach().clone())
        br = BasisRotationMatrices([p], lr=0.01, freq=10 ** 9)
        adam = torch.optim.Adam([q], lr=0.01)
        br.state[p].update(t=0, m=torch.zeros(5, 3), v=torch.zeros(5, 3), L=torch.zeros(5, 5), R=torch.zeros(3, 3),
                           U=torch.eye(5), V=torch.eye(3), refreshed=True)
        for _ in range(5):
            g = torch.randn(5, 3)
            p.grad, q.grad = g.clone(), g.clone()
            br.step()
            adam.step()
        torch.testing.assert_close(p.detach(), q.detach(), atol=1e-6, rtol=0)

    def test_harness_trains_every_parameter(self):
        trainer = make("B_cont_w4", 0, "stale", 4, 1 / 300)
        before = [q.clone() for q in trainer.net.parameters()]
        for k in range(12):
            trainer.step(torch.arange(32 * k, 32 * k + 32))
        for a, b in zip(before, trainer.net.parameters()):
            self.assertFalse(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
