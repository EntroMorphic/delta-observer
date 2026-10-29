"""Harness checks for research/reports/interface_confirm_protocol.md."""

import unittest

import torch

from research.experiments.contract_experiments import PermutedErrorTrainer
from research.experiments.interface_confirm import CENSOR, Warmup, build, paired_test, set_optimizer
from research.experiments.interface_sweep import Trainer


class InterfaceConfirmTests(unittest.TestCase):
    def test_warmup_ramps_then_holds(self):
        groups = [{"lr": 1.0}, {"lr": 0.5}]
        w = Warmup(groups, steps_per_epoch=2)  # 10 warmup steps
        self.assertEqual([g["lr"] for g in groups], [0.0, 0.0])
        for _ in range(5):
            w.step()
        self.assertAlmostEqual(groups[0]["lr"], 0.5)
        self.assertAlmostEqual(groups[1]["lr"], 0.25)
        for _ in range(20):
            w.step()
        self.assertEqual([g["lr"] for g in groups], [1.0, 0.5])

    def test_every_optimizer_moves_every_parameter(self):
        for name, lr in [("adam", 1e-3), ("sgd", 0.1), ("muon", 0.01)]:
            trainer = set_optimizer(Trainer("B_cont_w4", 0, 0), name, lr)
            before = [p.clone() for p in trainer.net.parameters()]
            for k in range(3):
                trainer.step(torch.arange(32 * k, 32 * k + 32))
            for p, q in zip(before, trainer.net.parameters()):
                self.assertFalse(torch.equal(p, q), name)

    def test_build_conditions(self):
        self.assertIsInstance(build("adder", "A_cont_w1", 0, "permuted", 16, "sgd", 0.1), PermutedErrorTrainer)
        fresh = build("automaton", "G_cont_w64", 0, "fresh", 16, "adam", 1e-3)
        self.assertEqual(fresh.delay, 0)
        self.assertEqual(len(fresh.inputs), 4096)
        self.assertEqual(build("adder", "A_cont_w1", 0, "stale", 16, "adam", 1e-3).delay, 16)

    def test_paired_test_censors_and_pairs(self):
        rows = [{"g": "a", "seed": s, "fit_epoch": 10} for s in range(8)]
        rows += [{"g": "b", "seed": s, "fit_epoch": None if s < 4 else 30} for s in range(8)]
        result = paired_test(rows, lambda r: r["g"] == "a", lambda r: r["g"] == "b", lambda r: r["seed"])
        self.assertEqual((result["n_pairs"], result["fits_better"], result["fits_worse"]), (8, 8, 4))
        self.assertEqual(result["wins"], 8)
        self.assertTrue(result["supported"])
        self.assertLess(result["p_one_sided"], 0.01)
        self.assertEqual(CENSOR, 201)


if __name__ == "__main__":
    unittest.main()
