"""Harness checks for research/reports/second_star_slip_v2_protocol.md."""

import copy
import unittest

import torch
import torch.nn.functional as F

from research.experiments.second_star_slip_v2 import Trainer, adder_dataset, predictors


def full_gradient(trainer, views=None):
    params = list(trainer.model.parameters())
    loss = F.mse_loss(trainer.model(trainer.inputs, views)[0], trainer.outputs)
    return torch.autograd.grad(loss, params, allow_unused=True)


class ChainAdderTests(unittest.TestCase):
    def test_dataset_is_binary_addition(self):
        x, y = adder_dataset(3)
        self.assertEqual(len(x), 2 ** 7)
        bits = lambda t, n: sum(int(t[i]) << i for i in range(n))
        for row, out in zip(x, y):
            self.assertEqual(bits(row[:3], 3) + bits(row[3:6], 3) + int(row[6]), bits(out, 4))

    def test_zero_delay_is_joint_training(self):
        a, b = Trainer(0, max_delay=4), Trainer(0, max_delay=4)
        for idx in a.epoch_batches(torch.Generator().manual_seed(1))[:6]:
            a.step(idx)
            b.step(idx, interface=2, delay=0)
        for p, q in zip(a.model.parameters(), b.model.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_stale_step_gives_each_side_the_other_side_delayed(self):
        trainer = Trainer(1, max_delay=4)
        batches = trainer.epoch_batches(torch.Generator().manual_seed(2))
        for idx in batches[:5]:
            trainer.step(idx)
        idx, old, live = batches[5], trainer.view(3), trainer.live()
        views = [live[m] if m <= 1 else old[m] for m in range(6)]
        loss = F.mse_loss(trainer.model(trainer.inputs[idx], views)[0], trainer.outputs[idx])
        params = [p for m in range(2) for p in live[m].values()]
        expected = [g.clone() for g in torch.autograd.grad(loss, params)]
        trainer.step(idx, interface=1, delay=3)
        for p, g in zip(params, expected):
            torch.testing.assert_close(p.grad, g)

    def test_cross_error_matches_exact_gradient_error_for_short_delay(self):
        trainer = Trainer(2, max_delay=2)
        for idx in trainer.epoch_batches(torch.Generator().manual_seed(3))[:40]:
            trainer.step(idx)
        j, old, live = 2, trainer.view(1), trainer.live()
        stale = []
        for side in (range(j + 1), range(j + 1, 6)):
            views = [live[m] if m in side else old[m] for m in range(6)]
            grads = full_gradient(trainer, views)
            names = [(m, n) for m in range(6) for n in live[m]]
            stale += [g for (m, _), g in zip(names, grads) if m in side]
        fresh = full_gradient(trainer)
        exact = torch.sqrt(sum(((s - f) ** 2).sum() for s, f in zip(stale, fresh)))
        row = predictors(trainer, 1)[j]
        self.assertAlmostEqual(row["secant_error"], float(exact), delta=1e-6 * float(exact) + 1e-9)

    def test_hessian_error_is_the_small_step_limit(self):
        # One real step leaves the linear regime (ReLU kinks); a tiny step does not.
        trainer = Trainer(2, max_delay=2)
        for idx in trainer.epoch_batches(torch.Generator().manual_seed(3))[:40]:
            trainer.step(idx)
        trainer.model.double()
        trainer.inputs, trainer.outputs = trainer.inputs.double(), trainer.outputs.double()
        now = trainer.snapshot()
        old = trainer.view(1)
        tiny = [{k: now[m][k] - 1e-4 * (now[m][k] - old[m][k].double()) for k in now[m]} for m in range(6)]
        trainer.history.extend([tiny, now])  # the "delayed" view is now a tiny step back
        row = predictors(trainer, 1)[2]
        self.assertLess(abs(row["hessian_error"] - row["secant_error"]) / row["secant_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
