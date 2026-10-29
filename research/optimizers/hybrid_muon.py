"""Muon hybrids with Adam-style second moments, for the staleness experiments.

PreconditionedMuon  divide the Nesterov momentum by Adam's sqrt(v) of the raw
                    gradient, then orthogonalize (precondition, then orthogonalize).
AdaMuonStyle        orthogonalize, then divide by a second moment of the
                    orthogonalized update and match Muon's RMS (a simplified
                    AdaMuon, arXiv 2507.11005: orthogonalize, then precondition).

Both treat every weight matrix this way; biases go to Adam. No weight decay.
"""

from __future__ import annotations

import torch

from .muon import zeroth_power_newton_schulz


def _shape_scale(p):
    return max(1.0, p.shape[0] / p.shape[1]) ** 0.5


class _HybridMatrices(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, beta2=0.999, eps=1e-8):
        super().__init__(params, dict(lr=lr, momentum=momentum, beta2=beta2, eps=eps))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError("matrix parameters only")

    def update(self, p, g, state, group):
        raise NotImplementedError

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["m"], state["v"], state["t"] = torch.zeros_like(p), torch.zeros_like(p), 0
                state["t"] += 1
                state["m"].mul_(group["momentum"]).add_(p.grad)
                nesterov = p.grad.add(state["m"], alpha=group["momentum"])
                p.add_(self.update(p, nesterov, state, group), alpha=-group["lr"])


class PreconditionedMuonMatrices(_HybridMatrices):
    def update(self, p, nesterov, state, group):
        b2, t = group["beta2"], state["t"]
        state["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
        v_hat = state["v"] / (1 - b2 ** t)
        return zeroth_power_newton_schulz(nesterov / (v_hat.sqrt() + group["eps"])) * _shape_scale(p)


class AdaMuonStyleMatrices(_HybridMatrices):
    def update(self, p, nesterov, state, group):
        b2, t = group["beta2"], state["t"]
        o = zeroth_power_newton_schulz(nesterov) * _shape_scale(p)
        state["v"].mul_(b2).addcmul_(o, o, value=1 - b2)
        u = o / ((state["v"] / (1 - b2 ** t)).sqrt() + group["eps"])
        return u * (o.norm() / u.norm().clamp_min(1e-12))


class MatricesPlusAdam:
    """A matrix optimizer for weight matrices; Adam for everything else."""

    def __init__(self, model, matrix_cls, lr, adam_lr):
        self.matrices = matrix_cls([p for p in model.parameters() if p.ndim == 2], lr=lr)
        self.adam = torch.optim.Adam([p for p in model.parameters() if p.ndim != 2], lr=adam_lr)
        self.param_groups = self.matrices.param_groups + self.adam.param_groups

    def step(self):
        self.matrices.step()
        self.adam.step()
