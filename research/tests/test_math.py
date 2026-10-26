import unittest
import platform

import torch

from models.delta_observer import generate_dataset
from research.experiments.optimizer_comparison import make_split, split_hash
from research.observers import ObserverD, ObserverDNeon
from research.optimizers.muon import HybridMuon, zeroth_power_newton_schulz


class StudentFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(9, 16)
        self.output = torch.nn.Linear(16, 5)


class MathTests(unittest.TestCase):
    def test_addition_truth_table(self):
        inputs, outputs, _, _ = generate_dataset()
        self.assertEqual(tuple(inputs.shape), (512, 9))
        for x, y in zip(inputs, outputs):
            a = sum(int(x[index]) << index for index in range(4))
            b = sum(int(x[index + 4]) << index for index in range(4))
            predicted = sum(int(y[index]) << index for index in range(5))
            self.assertEqual(predicted, a + b + int(x[8]))

    def test_split_is_disjoint_and_reproducible(self):
        first = make_split(7, 512)
        second = make_split(7, 512)
        combined = torch.cat(list(first.values()))
        self.assertEqual(len(torch.unique(combined)), 512)
        self.assertEqual(split_hash(first), split_hash(second))

    def test_newton_schulz_is_finite_and_shape_preserving(self):
        for shape in ((32, 64), (64, 32), (16, 16)):
            result = zeroth_power_newton_schulz(torch.randn(*shape))
            self.assertEqual(tuple(result.shape), shape)
            self.assertTrue(torch.isfinite(result).all())

    def test_hybrid_parameter_grouping(self):
        optimizer = HybridMuon(StudentFixture())
        self.assertIn("input.weight", optimizer.muon_names)
        self.assertIn("output.weight", optimizer.adamw_names)
        self.assertIn("input.bias", optimizer.adamw_names)

    def test_observer_d_native_parity(self):
        torch.manual_seed(11)
        encoder = torch.nn.Sequential(
            torch.nn.Linear(257, 128),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(128, 32),
        ).eval()
        features = torch.randn(19, 257)
        native = ObserverDNeon()
        actual = native.encode(features, encoder)
        with torch.no_grad():
            expected = encoder(features)
        self.assertLess(float((actual - expected).abs().max()), 1e-5)
        if platform.machine() in ("arm64", "aarch64"):
            self.assertEqual(native.backend, "c-neon")

    def test_observer_d_discovery_and_deployment_boundaries(self):
        torch.manual_seed(19)
        observer = ObserverD()
        source_a = torch.randn(7, 6, 64)
        source_b = torch.randn(7, 6, 64)
        inputs = torch.randn(7, 9)
        discovery = observer.discover(source_a, source_b)
        deployed = observer.deploy(inputs)
        self.assertEqual(tuple(discovery["method"].shape), (7, 16))
        self.assertEqual(tuple(discovery["behavior_logits"].shape), (7, 5))
        self.assertEqual(tuple(deployed["behavior_logits"].shape), (7, 5))
        self.assertTrue(torch.isfinite(observer.alignment_loss(
            deployed["method"], discovery["method"]
        )))

        exported = observer.export_deployment().eval()
        keys = tuple(exported.state_dict())
        self.assertFalse(any("source" in key or "temporal" in key for key in keys))
        with torch.no_grad():
            expected = observer.deploy(inputs)["behavior_logits"]
            actual = exported(inputs)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
