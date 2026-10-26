"""Part A of research/reports/curvature_protocol.md: optimizer identities."""

import unittest

import torch

from research.optimizers.muon import zeroth_power_newton_schulz

_previous_dtype = torch.get_default_dtype()


def setUpModule():
    torch.set_default_dtype(torch.float64)


def tearDownModule():
    torch.set_default_dtype(_previous_dtype)


def tiny_mlp(seed=0, n=8, d_in=3, hidden=4, d_out=2):
    """A tanh MLP as a function of one flat parameter vector."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_in, generator=generator)
    y = torch.randn(n, d_out, generator=generator)
    shapes = [(d_in, hidden), (hidden,), (hidden, d_out), (d_out,)]
    sizes = [torch.Size(s).numel() for s in shapes]

    def forward(w):
        w1, b1, w2, b2 = (p.reshape(s) for p, s in zip(torch.split(w, sizes), shapes))
        return torch.tanh(x @ w1 + b1) @ w2 + b2

    w = torch.randn(sum(sizes), generator=generator)
    return forward, w, y


def symmetric_power(matrix, power, tol=1e-12):
    """Matrix power on the non-null eigenspace (pseudo-inverse for negative powers)."""
    values, vectors = torch.linalg.eigh(matrix)
    keep = values > tol * values.max()
    scaled = torch.zeros_like(values)
    scaled[keep] = values[keep] ** power
    return vectors @ torch.diag(scaled) @ vectors.T


class HessianTests(unittest.TestCase):
    def test_a1_pearlmutter_hvp_is_exact(self):
        forward, w, y = tiny_mlp()
        loss = lambda p: ((forward(p) - y) ** 2).mean()
        hessian = torch.autograd.functional.hessian(loss, w)
        v = torch.randn_like(w)
        p = w.clone().requires_grad_(True)
        (gradient,) = torch.autograd.grad(loss(p), p, create_graph=True)
        (hvp,) = torch.autograd.grad(gradient @ v, p)
        torch.testing.assert_close(hvp, hessian @ v, rtol=1e-12, atol=1e-12)

    def test_a3_newton_is_attracted_to_saddles(self):
        # f(x, y) = x² − y²: gradient (2x, −2y), Hessian diag(2, −2)
        point = torch.tensor([0.7, -1.3])
        gradient = torch.tensor([2 * point[0], -2 * point[1]])
        hessian = torch.diag(torch.tensor([2.0, -2.0]))
        torch.testing.assert_close(point - torch.linalg.solve(hessian, gradient), torch.zeros(2))
        # With an indefinite Hessian the Newton direction can point uphill
        g = torch.tensor([1.0, 1.0])
        newton_direction = -torch.linalg.solve(torch.diag(torch.tensor([1.0, -0.5])), g)
        self.assertGreater(float(g @ newton_direction), 0.0)

    def test_a4_gauss_newton_is_psd_while_hessian_is_not(self):
        forward, w, y = tiny_mlp(seed=3)
        residual_count = y.numel()
        loss = lambda p: ((forward(p) - y) ** 2).mean()
        hessian = torch.autograd.functional.hessian(loss, w)
        f_flat = lambda p: forward(p).reshape(-1)
        jacobian = torch.autograd.functional.jacobian(f_flat, w)
        output_hessians = torch.stack([torch.autograd.functional.hessian(lambda p, i=i: f_flat(p)[i], w)
                                       for i in range(residual_count)])
        residual = f_flat(w) - y.reshape(-1)
        gauss_newton = 2 / residual_count * jacobian.T @ jacobian
        curvature_term = 2 / residual_count * torch.einsum("i,ijk->jk", residual, output_hessians)
        torch.testing.assert_close(hessian, gauss_newton + curvature_term, rtol=1e-10, atol=1e-12)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(gauss_newton).min()), -1e-12)
        self.assertLess(float(torch.linalg.eigvalsh(hessian).min()), 0.0)


class FirstOrderGeometryTests(unittest.TestCase):
    def test_a5_adam_without_averages_is_sign_descent(self):
        w = torch.randn(50, requires_grad=True)
        start = w.detach().clone()
        optimizer = torch.optim.Adam([w], lr=0.1, betas=(0.0, 0.0), eps=1e-30)
        w.grad = torch.randn(50)
        optimizer.step()
        torch.testing.assert_close(w.detach() - start, -0.1 * torch.sign(w.grad), rtol=1e-12, atol=1e-15)

    def test_a6_sign_is_steepest_descent_under_linf(self):
        g = torch.randn(20)
        best = float(g @ torch.sign(g))
        self.assertAlmostEqual(best, float(g.abs().sum()), places=12)
        candidates = torch.cat([torch.rand(5000, 20) * 2 - 1, torch.sign(torch.randn(5000, 20))])
        self.assertLessEqual(float((candidates @ g).max()), best + 1e-12)

    def test_a7_orthogonalized_gradient_is_steepest_under_spectral_norm(self):
        G = torch.randn(6, 4)
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        D_star = U @ Vh
        self.assertAlmostEqual(float(torch.linalg.matrix_norm(D_star, ord=2)), 1.0, places=12)
        best = float((G * D_star).sum())
        self.assertAlmostEqual(best, float(S.sum()), places=12)
        for _ in range(2000):
            D = torch.randn(6, 4)
            D = D / torch.linalg.matrix_norm(D, ord=2)
            self.assertLessEqual(float((G * D).sum()), best + 1e-12)


class MuonTests(unittest.TestCase):
    def setUp(self):
        self.G = torch.randn(64, 32, generator=torch.Generator().manual_seed(0))

    def test_a8_quintic_newton_schulz_keeps_singular_vectors_not_unit_values(self):
        out = zeroth_power_newton_schulz(self.G.float(), steps=5).double()
        U, _, Vh = torch.linalg.svd(self.G, full_matrices=False)
        projected = U.T @ out @ Vh.T  # diagonal iff out shares G's singular vectors
        off_diagonal = projected - torch.diag(torch.diagonal(projected))
        self.assertLess(float(off_diagonal.abs().max()), 1e-4)
        values = torch.linalg.svdvals(out)
        self.assertGreater(float(values.min()), 0.5)
        self.assertLess(float(values.max()), 1.5)
        self.assertGreater(float((values - 1).abs().max()), 0.05)  # not exactly orthogonal

    def test_a8_quintic_newton_schulz_leaves_tiny_singular_values_small(self):
        # Square random matrices have singular values near zero; five quintic
        # steps do not lift them to ~1, so Muon does not equalize every direction.
        G = torch.randn(256, 256, generator=torch.Generator().manual_seed(2))
        values = torch.linalg.svdvals(zeroth_power_newton_schulz(G.float(), steps=5).double())
        self.assertLess(float(values.min()), 0.1)
        self.assertLess(float(values.max()), 1.25)

    def test_a8_cubic_newton_schulz_converges_to_unit_singular_values(self):
        x = self.G / self.G.norm()
        for _ in range(60):
            x = 1.5 * x - 0.5 * x @ x.T @ x
        U, _, Vh = torch.linalg.svd(self.G, full_matrices=False)
        torch.testing.assert_close(x, U @ Vh, rtol=1e-10, atol=1e-10)

    def test_a9_shampoo_without_accumulation_is_orthogonalization(self):
        for shape in ((5, 5), (6, 4), (3, 7)):
            G = torch.randn(*shape)
            U, _, Vh = torch.linalg.svd(G, full_matrices=False)
            shampoo = symmetric_power(G @ G.T, -0.25) @ G @ symmetric_power(G.T @ G, -0.25)
            torch.testing.assert_close(shampoo, U @ Vh, rtol=1e-9, atol=1e-9)


class AdamWTests(unittest.TestCase):
    def test_a10_adamw_decay_is_decoupled(self):
        lr, wd, b1, b2, eps = 0.01, 0.1, 0.9, 0.999, 1e-8
        grads = [torch.randn(10, generator=torch.Generator().manual_seed(i)) for i in range(3)]
        w0 = torch.randn(10, generator=torch.Generator().manual_seed(99))

        def run(cls):
            w = w0.clone().requires_grad_(True)
            optimizer = cls([w], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
            for g in grads:
                w.grad = g.clone()
                optimizer.step()
            return w.detach()

        w, m, v = w0.clone(), torch.zeros(10), torch.zeros(10)
        for t, g in enumerate(grads, 1):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
            w = w - lr * (m_hat / (v_hat.sqrt() + eps) + wd * w)
        torch.testing.assert_close(run(torch.optim.AdamW), w, rtol=1e-12, atol=1e-14)
        self.assertGreater(float((run(torch.optim.Adam) - w).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
