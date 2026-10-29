"""Checks for research/experiments/compensation_bound.py."""

import unittest

import torch

from research.experiments.compensation_bound import make


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs])


class CompensationTests(unittest.TestCase):
    def test_taylor_is_accurate_for_a_small_controlled_movement(self):
        # others' parameters move by 1e-3 * noise from the stale snapshot; at this
        # scale few ReLU units flip, so first-order correction must remove most error
        errors = {}
        for c in ("stale", "taylor"):
            t = make("sgd", 0.1, "B_cont_w4", 0, c, 1)
            torch.manual_seed(1)
            with torch.no_grad():
                for p in t.net.parameters():
                    p.add_(1e-3 * torch.randn_like(p))
            x, y = t.inputs[:512], t.outputs[:512]
            g = flat(t.gradients(x, y))
            t.delay = 0
            errors[c] = float((g - flat(t.gradients(x, y))).norm())
        self.assertLess(errors["taylor"], 0.3 * errors["stale"])

    def test_compensations_vanish_without_movement(self):
        for c in ("dc_outer", "taylor"):
            t = make("sgd", 0.1, "A_cont_w1", 1, c, 4)
            s = make("sgd", 0.1, "A_cont_w1", 1, "stale", 4)
            x, y = t.inputs[:64], t.outputs[:64]
            for a, b in zip(t.gradients(x, y), s.gradients(x, y)):
                torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
