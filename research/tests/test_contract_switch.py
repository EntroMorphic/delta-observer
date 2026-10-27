"""Harness checks for research/experiments/contract_switch.py."""

import unittest

import torch

from research.experiments.contract_lifecycle import OneInterfaceTrainer
from research.experiments.contract_switch import SwitchTrainer, segments
from research.experiments.interface_sweep import Trainer


def advanced(cls, delay=4):
    trainer = cls("B_cont_w4", 3, delay)
    for k in range(6):
        trainer.step(torch.arange(16 * k, 16 * k + 16))
    return trainer


class ContractSwitchTests(unittest.TestCase):
    def test_segments(self):
        self.assertEqual(segments(6, set()), [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(segments(6, {1, 3}), [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(segments(6, set(range(5))), [[s] for s in range(6)])

    def test_all_decoupled_is_all_interface_staleness(self):
        switch, sweep = advanced(SwitchTrainer), advanced(Trainer)
        switch.decoupled = frozenset(range(5))
        x, y = switch.inputs[:64], switch.outputs[:64]
        for a, b in zip(switch.gradients(x, y), sweep.gradients(x, y)):
            torch.testing.assert_close(a, b)

    def test_one_decoupled_is_one_interface_staleness(self):
        switch, single = advanced(SwitchTrainer), advanced(OneInterfaceTrainer)
        switch.decoupled, single.interface = frozenset({2}), 2
        x, y = switch.inputs[:64], switch.outputs[:64]
        for a, b in zip(switch.gradients(x, y), single.gradients(x, y)):
            torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
