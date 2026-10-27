"""Hessian block structure of piecewise-linear networks (curvature report, addendum)."""

import unittest

import numpy as np
import torch
from torch import nn

from research.experiments.curvature_dynamics import exact_hessian, flat_loss
from research.experiments.curvature_structure import block_labels, gauss_newton, residual_split


def mlp(activation, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), activation(), nn.Linear(8, 8), activation(), nn.Linear(8, 3))


def split(model, seed=1):
    generator = torch.Generator().manual_seed(seed)
    x, y = torch.randn(32, 4, generator=generator), torch.randn(32, 3, generator=generator)
    labels, _ = block_labels(model)
    same = labels[:, None] == labels[None, :]
    loss, w = flat_loss(model, x, y)
    H, _ = exact_hessian(loss, w)
    return residual_split(H.numpy(), gauss_newton(model, x, w).numpy(), same)


class ResidualCurvatureTests(unittest.TestCase):
    def test_relu_residual_curvature_has_no_within_layer_part(self):
        # Inside one linear region the output is linear in each layer's weights
        # taken alone, so ∂²f/∂θ_l² = 0 and the residual term is zero on every
        # diagonal block: it is traceless and therefore indefinite.
        result = split(mlp(nn.ReLU))
        self.assertLess(result["within_residual"], 1e-20)
        self.assertLess(abs(result["residual_trace_relative"]), 1e-12)
        self.assertGreater(result["residual_negative"], 0)
        self.assertGreater(result["residual_positive"], 0)

    def test_smooth_activation_breaks_the_zero_blocks(self):
        self.assertGreater(split(mlp(nn.Tanh))["within_residual"], 1e-3)

    def test_block_labels_follow_top_level_modules(self):
        labels, names = block_labels(mlp(nn.ReLU))
        self.assertEqual(names, ["0", "2", "4"])
        np.testing.assert_array_equal(np.bincount(labels), [40, 72, 27])


if __name__ == "__main__":
    unittest.main()
