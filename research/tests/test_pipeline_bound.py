"""Checks for research/experiments/pipeline_bound.py."""

import unittest

import torch

from research.experiments.pipeline_bound import make


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs])


class PipelineBoundTests(unittest.TestCase):
    def test_stage_delays_decrease_to_zero(self):
        t = make("adam", 1 / 300, "B_cont_w4", 0, "stale", 10)
        self.assertEqual(t.stage_delays(), [10, 8, 6, 4, 2, 0])

    def test_pipeline_gradient_uses_each_stage_snapshot(self):
        t = make("adam", 1 / 300, "B_cont_w4", 0, "stale", 5)
        for k in range(8):
            t.step(torch.arange(32 * k, 32 * k + 32))
        x, y = t.inputs[:64], t.outputs[:64]
        got = t.gradients(x, y)
        offset = 0
        for k, tau in enumerate(t.stage_delays()):
            snap = t.view(tau)
            own = {n: p.clone().requires_grad_(True) for n, p in snap[k].items()}
            views = [own if s == k else snap[s] for s in range(t.net.n)]
            if tau == 0:
                live = [{n: p.detach() for n, p in s.items()} for s in t.live()]
                own = {n: p.clone().requires_grad_(True) for n, p in live[k].items()}
                views = [own if s == k else live[s] for s in range(t.net.n)]
            want = torch.autograd.grad(t.full_loss(views, x, y), list(own.values()))
            for a, b in zip(want, got[offset:offset + len(want)]):
                torch.testing.assert_close(a, b)
            offset += len(want)

    def test_zero_history_is_fresh_and_permuted_keeps_entries(self):
        stale = make("basis_rotation", 1 / 300, "F_pipeline_w64", 0, "stale", 10)
        fresh = make("basis_rotation", 1 / 300, "F_pipeline_w64", 0, "fresh", 0)
        x, y = stale.inputs[:64], stale.outputs[:64]
        torch.testing.assert_close(flat(stale.gradients(x, y)), flat(fresh.gradients(x, y)))
        perm = make("adam", 1 / 300, "B_cont_w4", 1, "permuted", 5)
        st = make("adam", 1 / 300, "B_cont_w4", 1, "stale", 5)
        for t in (perm, st):
            for k in range(8):
                t.step(torch.arange(32 * k, 32 * k + 32))
        # identical trajectories so far? only if errors were zero; instead check entries on one batch
        f = make("adam", 1 / 300, "B_cont_w4", 1, "fresh", 0)
        f.net.load_state_dict(st.net.state_dict())
        x, y = st.inputs[:64], st.outputs[:64]
        s_err = [a - b for a, b in zip(st.gradients(x, y), f.gradients(x, y))]
        perm.net.load_state_dict(st.net.state_dict())
        perm.history = st.history
        p_err = [a - b for a, b in zip(perm.gradients(x, y), f.gradients(x, y))]
        for a, b in zip(s_err, p_err):
            torch.testing.assert_close(a.reshape(-1).sort().values, b.reshape(-1).sort().values)


if __name__ == "__main__":
    unittest.main()
