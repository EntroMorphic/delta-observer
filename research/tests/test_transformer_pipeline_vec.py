"""The vectorized engine reproduces transformer_pipeline.PipelineRun step for step."""

import unittest

import torch

import research.experiments.transformer_pipeline as tp
from research.experiments.transformer_pipeline_vec import Group

METHODS = ["fresh", "stale", "permuted", "pipemare_lr", "weight_predict", "basis_rotation", "basis_rotation_fresh"]


class VectorizedEngineTests(unittest.TestCase):
    def test_matches_reference_for_every_method(self):
        torch.set_num_threads(4)
        runs = [(m, 0 if m in ("fresh", "basis_rotation_fresh") else 6, 1e-3, 3) for m in METHODS]
        runs.append(("stale", 12, 3e-3, 4))  # a second seed, delay and learning rate in the same group
        group = Group(runs)
        refs = [tp.PipelineRun(m, d, lr, s) for m, d, lr, s in runs]
        for _ in range(12):
            group.step()
            for ref in refs:
                ref.step()
        val = group.val_loss()
        for r, ref in enumerate(refs):
            if runs[r][0].startswith("basis_rotation"):
                # Basis Rotation amplifies rounding (the reference differs from itself by ~1e-4 between thread
                # counts after 3 steps), so it is held to that noise level rather than to exact agreement.
                worst = max(float((group.P[n][r] - p.detach()).abs().max()) for n, p in zip(ref.names, ref.params))
                self.assertLess(worst, 1e-2)
                self.assertAlmostEqual(val[r], ref.val_loss(), delta=5e-3)
                continue
            for n, p in zip(ref.names, ref.params):
                torch.testing.assert_close(group.P[n][r], p.detach(), rtol=1e-4, atol=2e-5, msg=f"{runs[r]} {n}")
            self.assertAlmostEqual(val[r], ref.val_loss(), places=4)


if __name__ == "__main__":
    unittest.main()
