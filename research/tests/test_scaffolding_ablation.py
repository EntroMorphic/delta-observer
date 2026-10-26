import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import models.delta_observer as paper
from research.experiments import scaffolding_ablation as ablation


class ScaffoldingAblationTests(unittest.TestCase):
    def test_paper_config_replicates_paper_pipeline_exactly(self):
        epochs, seed = 22, 3
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.multiple(paper, DATA_DIR=tmp, MODELS_DIR=tmp, LOGS_DIR=tmp / "logs"), \
                    contextlib.redirect_stdout(io.StringIO()):
                paper.train_online(epochs=epochs, seed=seed)
            reference = np.load(tmp / "online_observer_trajectory.npz")["snapshots"]
            reference_final = np.load(tmp / "online_observer_latents.npz")["latents"]

        replica = ablation.train(ablation.PAPER, seed, epochs, snapshots="replicate")
        self.assertEqual(len(replica["snapshots"]), len(reference))
        for ours, theirs in zip(replica["snapshots"], reference):
            np.testing.assert_array_equal(ours, theirs)
        np.testing.assert_array_equal(replica["final_latent"], reference_final)

    def test_snapshots_do_not_perturb_training(self):
        config = ablation.configurations()["N/observer"]
        with mock.patch.object(ablation, "PCA") as pca:
            pca.return_value.fit_transform.side_effect = lambda x: x[:, :16]
            measured = ablation.train(config, seed=1, epochs=3)
        torch.manual_seed(0)  # unrelated global state must not matter either
        replica = ablation.train(config, seed=1, epochs=3, snapshots="none")
        sil_final, r2_final = ablation.latent_metrics(replica["final_latent"],
                                                      ablation.make_data(config["data"])[2].numpy())
        self.assertAlmostEqual(measured["summary"]["online_sil"], sil_final, places=12)
        self.assertAlmostEqual(measured["summary"]["online_r2"], r2_final, places=12)

    def test_each_flip_changes_exactly_one_factor(self):
        configs = ablation.configurations()
        self.assertEqual(len(configs), 2 + 2 * len(ablation.FACTORS))
        for name, config in configs.items():
            if "/" not in name:
                continue
            base, factor = name.split("/")
            baseline = ablation.PAPER if base == "P" else ablation.NOTEBOOK
            changed = [f for f in ablation.FACTORS if config[f] != baseline[f]]
            self.assertEqual(changed, [factor])

    def test_addendum_a_splits_the_observer_factor(self):
        configs = ablation.addendum_a_configurations()
        for name in ("P/obs-nobit", "P/obs-constlr"):
            changed = [f for f in ablation.FACTORS if configs[name][f] != ablation.PAPER[f]]
            self.assertEqual(changed, ["observer"])
        result = ablation.train(configs["P/obs-nobit"], seed=0, epochs=2)
        for measure in ablation.SOURCE_MEASURES:
            self.assertEqual(len(result["history"][f"sil_{measure}"]), 2)
            self.assertIn("present", result["summary"][measure])

    def test_unsupervised_observer_has_no_supervised_heads(self):
        config = ablation.addendum_b_configurations()["P/obs-unsup"]
        self.assertEqual([f for f in ablation.FACTORS if config[f] != ablation.PAPER[f]], ["observer"])
        observer = ablation.Observer(bit_head=False, carry_head=False)
        self.assertIsNone(observer.bit_classifier)
        self.assertIsNone(observer.carry_regressor)
        result = ablation.train(config, seed=0, epochs=2)
        self.assertEqual(len(result["history"]["r2_src_pca16"]), 2)

    def test_simplified_data_is_carry_free_addition(self):
        inputs, outputs, carry_counts, _ = ablation.make_data("256")
        self.assertEqual(tuple(inputs.shape), (256, 8))
        for x, y in zip(inputs, outputs):
            a = sum(int(x[i]) << i for i in range(4))
            b = sum(int(x[i + 4]) << i for i in range(4))
            self.assertEqual(sum(int(y[i]) << i for i in range(5)), a + b)
        self.assertEqual(int(carry_counts.max()), 4)

    def test_all_configurations_train(self):
        for name, config in ablation.configurations().items():
            with self.subTest(name):
                result = ablation.train(config, seed=0, epochs=2)
                self.assertEqual(len(result["history"]["sil_eval"]), 2)
                self.assertTrue(np.isfinite(result["summary"]["online_r2"]))


if __name__ == "__main__":
    unittest.main()
