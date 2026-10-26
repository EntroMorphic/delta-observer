"""A small, dependency-free Muon implementation for controlled experiments.

Muon applies momentum followed by a Newton--Schulz orthogonalization to matrix
updates.  Vectors, scalars, and the student output head are optimized by AdamW
through :class:`HybridMuon`; this is the usual "Muon hybrid" setup rather than
an assertion that Muon is suitable for every parameter tensor.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def zeroth_power_newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate ``U @ V.T`` for a matrix with SVD ``U @ S @ V.T``.

    The polynomial coefficients are the standard quintic Muon coefficients.
    Computation stays in float32 so this also works reliably on CPU and MPS.
    """
    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization requires a 2-D tensor")

    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    x = x / x.norm().clamp_min(1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x

    if transposed:
        x = x.T
    return x.to(dtype=matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for two-dimensional parameters only."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("Muon accepts matrix parameters only")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.mul_(group["momentum"]).add_(gradient)
                update = gradient.add(buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
                update = zeroth_power_newton_schulz(update, group["ns_steps"])

                # Preserve update RMS across rectangular matrix shapes.
                update.mul_(max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5)
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


class HybridMuon:
    """Muon for hidden matrices and AdamW for all remaining parameters."""

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float = 0.02,
        adamw_lr: float = 3e-3,
        weight_decay: float = 0.01,
    ) -> None:
        muon_params = []
        adamw_params = []
        self.muon_names = []
        self.adamw_names = []
        for name, parameter in model.named_parameters():
            # Output mappings and non-matrices use AdamW. Muon handles the
            # representation-learning matrices for which it was designed.
            if parameter.ndim == 2 and not name.startswith("output"):
                muon_params.append(parameter)
                self.muon_names.append(name)
            else:
                adamw_params.append(parameter)
                self.adamw_names.append(name)

        self.muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
        self.adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay)
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    @property
    def state(self):
        return {**self.muon.state, **self.adamw.state}
