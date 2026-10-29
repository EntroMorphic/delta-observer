"""Basis Rotation (Jung, Shin & Lee, arXiv 2602.03515), small-scale reimplementation.

Per weight matrix: Kronecker-factored second moments L = EMA(G G^T) and
R = EMA(G^T G) (beta2), refreshed every `freq` steps; the eigenbases U, V are
updated by one power step and a QR decomposition (bilateral, second-moment
source: the paper's defaults). Adam runs in the rotated basis and the update
is rotated back. Unstated in the paper, chosen here: U and V start from an
exact eigendecomposition at the first refresh (as in SOAP), and the rotated
second moment is not reset when the basis changes. Biases go to Adam.
"""

from __future__ import annotations

import torch


class BasisRotationMatrices(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, freq=10):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, freq=freq))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError("matrix parameters only")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if not st:
                    m, n = p.shape
                    st.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p), L=torch.zeros(m, m),
                              R=torch.zeros(n, n), U=torch.eye(m), V=torch.eye(n), refreshed=False)
                st["t"] += 1
                st["m"].mul_(b1).add_(g, alpha=1 - b1)
                if st["t"] % group["freq"] == 0 or not st["refreshed"]:
                    st["L"].mul_(b2).add_(g @ g.T, alpha=1 - b2)
                    st["R"].mul_(b2).add_(g.T @ g, alpha=1 - b2)
                    if not st["refreshed"]:
                        st["U"] = torch.linalg.eigh(st["L"])[1].flip(1)
                        st["V"] = torch.linalg.eigh(st["R"])[1].flip(1)
                        st["refreshed"] = True
                    else:
                        st["U"] = torch.linalg.qr(st["L"] @ st["U"])[0]
                        st["V"] = torch.linalg.qr(st["R"] @ st["V"])[0]
                U, V = st["U"], st["V"]
                g_rot, m_rot = U.T @ g @ V, U.T @ st["m"] @ V
                st["v"].mul_(b2).addcmul_(g_rot, g_rot, value=1 - b2)
                m_hat = m_rot / (1 - b1 ** st["t"])
                v_hat = st["v"] / (1 - b2 ** st["t"])
                p.add_(U @ (m_hat / (v_hat.sqrt() + group["eps"])) @ V.T, alpha=-group["lr"])


class BasisRotationAdam:
    """Basis Rotation for weight matrices; Adam for biases."""

    def __init__(self, model, lr):
        self.matrices = BasisRotationMatrices([p for p in model.parameters() if p.ndim == 2], lr=lr)
        self.adam = torch.optim.Adam([p for p in model.parameters() if p.ndim != 2], lr=lr)
        self.param_groups = self.matrices.param_groups + self.adam.param_groups

    def step(self):
        self.matrices.step()
        self.adam.step()
