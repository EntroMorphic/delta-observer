import unittest

import torch

from models.delta_observer import generate_dataset
from research.experiments import curvature_dynamics as cd
from research.experiments.scaffolding_ablation import PaperComp


class CurvatureDynamicsTests(unittest.TestCase):
    def test_exact_hessian_matches_reference(self):
        torch.manual_seed(0)
        inputs, outputs, _, _ = generate_dataset()
        loss, w = cd.flat_loss(PaperComp(), inputs, outputs)
        H, asymmetry = cd.exact_hessian(loss, w)
        torch.testing.assert_close(H, torch.autograd.functional.hessian(loss, w), rtol=1e-10, atol=1e-12)
        self.assertLess(asymmetry, 1e-12)
        self.assertAlmostEqual(cd.sharpness(loss, w), float(torch.linalg.eigvalsh(H)[-1]), places=8)

    def test_analysis_on_known_spectrum(self):
        H = torch.diag(torch.tensor([3.0, 2.0, -1.0, 0.5], dtype=torch.float64))
        g = torch.tensor([1.0, 0.0, 2.0, 0.0], dtype=torch.float64)
        metrics, top5 = cd.analyze(H, g)
        self.assertEqual(metrics["negative_count"], 1)
        self.assertAlmostEqual(metrics["grad_fraction_topk"]["1"], 1 / 5)
        # g·(−H⁻¹g) = −(1/3 + 4/(−1)) = 11/3 > 0: descent despite the negative eigenvalue
        self.assertTrue(metrics["newton_is_descent"])
        self.assertAlmostEqual(metrics["newton_negative_curvature_share"], 4 / (4 + 1 / 3))
        metrics2, _ = cd.analyze(H, torch.tensor([3.0, 0, 1.0, 0], dtype=torch.float64), top5)
        self.assertFalse(metrics2["newton_is_descent"])  # −(9/3 + 1/(−1)) = −2
        self.assertAlmostEqual(metrics2["top5_overlap_with_previous"], 1.0)


if __name__ == "__main__":
    unittest.main()
