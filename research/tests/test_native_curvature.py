import unittest
import torch

from research.lfm.audit_native_curvature import directional_derivatives


class CurvatureTests(unittest.TestCase):
    def test_quadratic_known_hessian(self):
        x=torch.tensor([2.,3.],requires_grad=True)
        direction=torch.tensor([.6,.8])
        loss=2*x[0]**2+3*x[1]**2
        first,second=directional_derivatives(loss,[x],[direction])
        self.assertAlmostEqual(first,19.2,places=5)
        self.assertAlmostEqual(second,5.28,places=5)
