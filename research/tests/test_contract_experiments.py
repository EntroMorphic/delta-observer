"""Harness checks for research/experiments/contract_experiments.py."""

import unittest

import numpy as np
import torch

from research.experiments.contract_experiments import (
    AUTOMATON_HIDDEN, N_STATES, NoiseTrainer, Trainer, automaton_dataset, automaton_states, participation_ratio)


class ContractExperimentTests(unittest.TestCase):
    def test_automaton_targets_follow_the_machine(self):
        x, y = automaton_dataset(5)
        self.assertEqual(tuple(x.shape), (4096, 13))
        states = automaton_states(5, x)
        rng = np.random.default_rng(5)
        delta = rng.integers(N_STATES, size=(N_STATES, 4))
        for i in [0, 17, 999, 4095]:
            symbols = [int(x[i, s] + 2 * x[i, s + 6]) for s in range(6)]
            # the label after stage s equals the label of the state entering stage s + 1
            for s in range(5):
                self.assertEqual(delta[states[i, s], symbols[s]], states[i, s + 1])
        same_state = {}
        for i in range(len(x)):
            for s in range(1, 6):
                same_state.setdefault(int(states[i, s]), set()).add(int(y[i, s - 1]))
        self.assertTrue(all(len(v) == 1 for v in same_state.values()))

    def test_noise_matches_staleness_error_norm_per_tensor(self):
        noisy, stale, fresh = NoiseTrainer("A_cont_w1", 0, 2), Trainer("A_cont_w1", 0, 2), Trainer("A_cont_w1", 0, 0)
        for t in (noisy, stale, fresh):
            for k in range(3):
                t.history.append([{n: p + 0.01 * (k + 1) for n, p in v.items()} for v in t.history[-1]])
        x, y = noisy.inputs[:64], noisy.outputs[:64]
        for n, s, f in zip(noisy.gradients(x, y), stale.gradients(x, y), fresh.gradients(x, y)):
            torch.testing.assert_close((n - f).norm(), (s - f).norm())

    def test_permuted_error_keeps_entries_and_norm(self):
        from research.experiments.contract_experiments import PermutedErrorTrainer
        perm, stale, fresh = PermutedErrorTrainer("A_cont_w1", 0, 2), Trainer("A_cont_w1", 0, 2), Trainer("A_cont_w1", 0, 0)
        for t in (perm, stale, fresh):
            for k in range(3):
                t.history.append([{n: p + 0.01 * (k + 1) for n, p in v.items()} for v in t.history[-1]])
        x, y = perm.inputs[:64], perm.outputs[:64]
        for p, s, f in zip(perm.gradients(x, y), stale.gradients(x, y), fresh.gradients(x, y)):
            torch.testing.assert_close((p - f).reshape(-1).sort().values, (s - f).reshape(-1).sort().values)

    def test_correlated_noise_has_set_size_and_memory(self):
        from research.experiments.contract_experiments import CorrelatedNoiseTrainer
        noisy, fresh = CorrelatedNoiseTrainer("A_cont_w1", 0, 0), Trainer("A_cont_w1", 0, 0)
        x, y = noisy.inputs[:64], noisy.outputs[:64]
        f = fresh.gradients(x, y)
        first = [n - g for n, g in zip(noisy.gradients(x, y), f)]
        second = [n - g for n, g in zip(noisy.gradients(x, y), f)]
        for a, g in zip(first, f):
            torch.testing.assert_close(a.norm(), 0.3 * g.norm())
        cos = torch.nn.functional.cosine_similarity(first[2].reshape(-1), second[2].reshape(-1), dim=0)
        self.assertGreater(float(cos), 0.5)  # rho = 0.75 for tau = 4

    def test_split_is_disjoint_and_covers(self):
        from research.experiments.contract_experiments import split
        from research.experiments.second_star_slip_v2 import adder_dataset
        (tx, _), (vx, _) = split(adder_dataset(), 3)
        self.assertEqual(len(tx) + len(vx), 8192)
        rows = {tuple(r.tolist()) for r in tx}
        self.assertFalse(any(tuple(r.tolist()) in rows for r in vx[:500]))

    def test_automaton_arms_build(self):
        data = automaton_dataset(1)
        for arm in ["A_cont_w1", "G_cont_w64", "F_pipeline_w64"]:
            Trainer(arm, 1, 4, 0, data, AUTOMATON_HIDDEN).step(torch.arange(32))

    def test_two_means_finds_the_gap(self):
        from research.experiments.contract_experiments import two_means
        values = torch.tensor([-0.4, -0.35, -0.3, 0.9, 1.0, 1.1])
        threshold, low, high = two_means(values)
        self.assertAlmostEqual(low, -0.35, places=5)
        self.assertAlmostEqual(high, 1.0, places=5)
        self.assertTrue(-0.3 < threshold < 0.9)

    def test_snapped_messages_are_two_level_and_carry_no_gradient(self):
        from research.experiments.contract_experiments import two_level_snap
        trainer = Trainer("A_cont_w1", 0, 0)
        trainer.net.snap = two_level_snap(trainer)
        live = trainer.live()
        _, _, message = trainer.net.stage(0, live[0], trainer.inputs, trainer.net.initial_message(trainer.inputs))
        self.assertLessEqual(len(message.unique()), 2)
        self.assertFalse(message.requires_grad)

    def test_participation_ratio(self):
        m = torch.randn(4000, 3) * torch.tensor([1.0, 1.0, 0.0])
        self.assertAlmostEqual(participation_ratio([None, m])[0], 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
