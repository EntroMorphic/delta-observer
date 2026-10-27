"""Harness checks for research/reports/second_star_slip_protocol.md."""

import unittest

import torch
import torch.nn.functional as F

from research.experiments.curvature_dynamics import train_sources
from research.experiments.second_star_slip import Trainer, batches, delay_matrix, mixed_forward, snapshot


class StalenessHarnessTests(unittest.TestCase):
    def test_delays_add_along_the_chain(self):
        delays = delay_matrix((1, 10, 100))
        self.assertEqual(delays[0][1], 1)
        self.assertEqual(delays[1][2], 10)
        self.assertEqual(delays[0][3], 111)
        self.assertEqual(delays[3][1], 110)
        self.assertEqual([delays[m][m] for m in range(4)], [0, 0, 0, 0])

    def test_fresh_trainer_reproduces_curvature_harness(self):
        states, _ = train_sources(0, epochs=2)
        trainer = Trainer(0, epochs=2)
        for _ in range(2):
            for idx in batches(trainer):
                trainer.step(idx)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, states[("comp", 2)][name], rtol=0, atol=0)

    def test_split_path_with_zero_delay_matches_joint_training(self):
        joint, split = Trainer(1, epochs=1), Trainer(1, epochs=1)
        torch.manual_seed(5)
        for idx in batches(joint):
            joint.step(idx)
            split.step(idx, force_split=True)
        for a, b in zip(joint.model.parameters(), split.model.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_mixed_forward_matches_model_at_current_parameters(self):
        trainer = Trainer(2)
        with torch.no_grad():
            torch.testing.assert_close(mixed_forward(trainer.model, snapshot(trainer.model), trainer.inputs),
                                       trainer.model(trainer.inputs))

    def test_stale_gradient_uses_old_downstream_parameters(self):
        trainer = Trainer(3)
        for idx in batches(trainer)[:5]:
            trainer.step(idx)
        idx = batches(trainer)[0]
        old, live = trainer.view(3), snapshot(trainer.model)
        views = [{n: p.clone().requires_grad_(True) for n, p in live[0].items()}, old[1], old[2], old[3]]
        loss = F.mse_loss(mixed_forward(trainer.model, views, trainer.inputs[idx]), trainer.outputs[idx])
        expected = torch.autograd.grad(loss, list(views[0].values()))
        trainer.step(idx, ks=(3, 0, 0))  # module 0 sees modules 1-3 as they were 3 steps ago
        for p, g in zip(trainer.model.bit0.parameters(), expected):
            torch.testing.assert_close(p.grad, g)


if __name__ == "__main__":
    unittest.main()
