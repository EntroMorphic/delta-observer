"""Fundamental Observer.D discovery and deployment architecture."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObserverDDeployment(nn.Module):
    """The independently executable third model retained after discovery."""

    def __init__(self, deployment_encoder: nn.Module, behavior_decoder: nn.Module):
        super().__init__()
        # Copying makes source detachment structural rather than conventional.
        self.deployment_encoder = copy.deepcopy(deployment_encoder)
        self.behavior_decoder = copy.deepcopy(behavior_decoder)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        method = self.deployment_encoder(inputs)
        return self.behavior_decoder(method)


class ObserverD(nn.Module):
    """One model containing discovery-time and deployment-time paths.

    The temporal path discovers a method representation from paired source
    trajectories. The deployment path is present from initialization and learns
    to enter that method space directly from the original task input.
    """

    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 5,
        source_a_dim: int = 64,
        source_b_dim: int = 64,
        adapter_dim: int = 24,
        method_dim: int = 16,
    ) -> None:
        super().__init__()
        self.source_a_adapter = nn.Sequential(
            nn.Linear(source_a_dim, adapter_dim), nn.GELU(approximate="tanh")
        )
        self.source_b_adapter = nn.Sequential(
            nn.Linear(source_b_dim, adapter_dim), nn.GELU(approximate="tanh")
        )
        # A/B state, A/B delta, and normalized time.
        temporal_dim = 4 * adapter_dim + 1
        self.temporal_method_extractor = nn.GRU(
            input_size=temporal_dim,
            hidden_size=method_dim,
            batch_first=True,
        )
        self.deployment_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(approximate="tanh"),
            nn.Linear(32, method_dim),
        )
        self.behavior_decoder = nn.Sequential(
            nn.Linear(method_dim, 16),
            nn.GELU(approximate="tanh"),
            nn.Linear(16, output_dim),
        )
        self.source_a_decoder = nn.Linear(method_dim, source_a_dim)
        self.source_b_decoder = nn.Linear(method_dim, source_b_dim)
        self.method_dim = method_dim

    @staticmethod
    def _delta(trajectory: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(trajectory)
        delta[:, 1:] = trajectory[:, 1:] - trajectory[:, :-1]
        return delta

    def discover(
        self,
        source_a_trajectory: torch.Tensor,
        source_b_trajectory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Discover a method from ``[batch, time, feature]`` trajectories."""
        if source_a_trajectory.ndim != 3 or source_b_trajectory.ndim != 3:
            raise ValueError("Observer.D trajectories must have [batch, time, feature] shape")
        if source_a_trajectory.shape[:2] != source_b_trajectory.shape[:2]:
            raise ValueError("Source trajectories must share batch and time dimensions")

        state_a = self.source_a_adapter(source_a_trajectory)
        state_b = self.source_b_adapter(source_b_trajectory)
        delta_a = self.source_a_adapter(self._delta(source_a_trajectory))
        delta_b = self.source_b_adapter(self._delta(source_b_trajectory))
        batch, steps = source_a_trajectory.shape[:2]
        time = torch.linspace(
            0.0,
            1.0,
            steps,
            dtype=source_a_trajectory.dtype,
            device=source_a_trajectory.device,
        ).view(1, steps, 1).expand(batch, -1, -1)
        sequence = torch.cat((state_a, state_b, delta_a, delta_b, time), dim=-1)
        temporal_states, hidden = self.temporal_method_extractor(sequence)
        method = hidden[-1]
        return {
            "method": method,
            "behavior_logits": self.behavior_decoder(method),
            "source_a_reconstruction": self.source_a_decoder(temporal_states),
            "source_b_reconstruction": self.source_b_decoder(temporal_states),
        }

    def deploy(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        method = self.deployment_encoder(inputs)
        return {"method": method, "behavior_logits": self.behavior_decoder(method)}

    def alignment_loss(
        self, deployed_method: torch.Tensor, discovered_method: torch.Tensor
    ) -> torch.Tensor:
        """Align deployment to a fixed discovery target without moving it."""
        deployed = F.normalize(deployed_method, dim=-1)
        discovered = F.normalize(discovered_method.detach(), dim=-1)
        return F.mse_loss(deployed, discovered)

    def export_deployment(self) -> ObserverDDeployment:
        """Physically copy only components needed for independent inference."""
        return ObserverDDeployment(self.deployment_encoder, self.behavior_decoder)
