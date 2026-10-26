"""Compact, weight-shared deployment substrate for Observer.D-LM.

This is a candidate substrate to receive the method discovered from LFM2.5,
not evidence that its architecture has already been observationally derived.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ObserverDLMConfig:
    vocab_size: int = 65_536
    width: int = 384
    repetitions: int = 8
    attention_every: int = 2
    attention_heads: int = 6
    expansion: int = 4
    local_kernel: int = 3
    max_context: int = 2_048
    dropout: float = 0.0
    output_head_rank: int = 0
    output_bias: bool = False
    attention_residual_scale: float = 1.0
    ffn_residual_scale: float = 1.0
    depth_adapter_rank: int = 0
    input_embedding_rank: int = 0
    method_cell_count: int = 1


class SharedMethodCell(nn.Module):
    """One local/global method operator reused across model depth."""

    def __init__(self, config: ObserverDLMConfig):
        super().__init__()
        width = config.width
        self.local_norm = nn.RMSNorm(width)
        self.local_gate = nn.Linear(width, 2 * width, bias=False)
        self.local_conv = nn.Conv1d(
            width,
            width,
            kernel_size=config.local_kernel,
            groups=width,
            bias=False,
        )
        self.local_output = nn.Linear(width, width, bias=False)
        self.attention_norm = nn.RMSNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
            bias=False,
        )
        self.ffn_norm = nn.RMSNorm(width)
        hidden = config.expansion * width
        self.ffn_gate = nn.Linear(width, hidden, bias=False)
        self.ffn_up = nn.Linear(width, hidden, bias=False)
        self.ffn_down = nn.Linear(hidden, width, bias=False)
        self.kernel = config.local_kernel
        self.attention_residual_scale = config.attention_residual_scale
        self.ffn_residual_scale = config.ffn_residual_scale

    def local(self, states: torch.Tensor) -> torch.Tensor:
        normalized = self.local_norm(states)
        gate, value = self.local_gate(normalized).chunk(2, dim=-1)
        mixed = F.silu(gate) * value
        mixed = F.pad(mixed.transpose(1, 2), (self.kernel - 1, 0))
        mixed = self.local_conv(mixed).transpose(1, 2)
        return states + self.local_output(mixed)

    def global_attention(self, states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(states)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
            is_causal=True,
        )
        return states + self.attention_residual_scale * attended

    def feed_forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = self.ffn_norm(states)
        update = F.silu(self.ffn_gate(normalized)) * self.ffn_up(normalized)
        return states + self.ffn_residual_scale * self.ffn_down(update)


class ObserverDLM(nn.Module):
    """Source-free language model exported from the Observer.D process."""

    def __init__(self, config: ObserverDLMConfig = ObserverDLMConfig()):
        super().__init__()
        if config.width % config.attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if not 1 <= config.method_cell_count <= config.repetitions:
            raise ValueError("method_cell_count must be between 1 and repetitions")
        if config.input_embedding_rank and not config.output_head_rank:
            raise ValueError("Factorized input embeddings require a separate output head")
        self.config = config
        input_width = config.input_embedding_rank or config.width
        self.token_embedding = nn.Embedding(config.vocab_size, input_width)
        self.input_projection = (
            nn.Linear(input_width, config.width, bias=False)
            if config.input_embedding_rank else None
        )
        self.depth_embedding = nn.Parameter(torch.zeros(config.repetitions, config.width))
        self.method_cell = SharedMethodCell(config)
        self.additional_method_cells = nn.ModuleList([
            SharedMethodCell(config) for _ in range(config.method_cell_count - 1)
        ])
        self.final_norm = nn.RMSNorm(config.width)
        if config.depth_adapter_rank < 0:
            raise ValueError("depth_adapter_rank must be non-negative")
        self.depth_adapters = nn.ModuleList([
            nn.Sequential(
                nn.RMSNorm(config.width),
                nn.Linear(config.width, config.depth_adapter_rank, bias=False),
                nn.SiLU(),
                nn.Linear(config.depth_adapter_rank, config.width, bias=False),
            )
            for _ in range(config.repetitions)
        ]) if config.depth_adapter_rank else nn.ModuleList()
        if config.output_head_rank < 0:
            raise ValueError("output_head_rank must be non-negative")
        if config.output_head_rank:
            self.output_projection = nn.Linear(
                config.width, config.output_head_rank, bias=False
            )
            self.output_embedding = nn.Embedding(
                config.vocab_size, config.output_head_rank
            )
        else:
            self.output_projection = None
            self.output_embedding = None
        self.lm_head_bias = (
            nn.Parameter(torch.zeros(config.vocab_size)) if config.output_bias else None
        )
        self.apply(self._initialize_module)
        for method_cell in (self.method_cell, *self.additional_method_cells):
            nn.init.normal_(method_cell.attention.in_proj_weight, mean=0.0, std=0.02)
        for adapter in self.depth_adapters:
            nn.init.zeros_(adapter[-1].weight)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    @property
    def output_weight(self) -> torch.Tensor:
        """The language head is tied to the input embedding."""
        if self.output_embedding is not None:
            raise RuntimeError("Low-rank output heads do not expose one resident dense weight")
        return self.token_embedding.weight

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(input_ids)
        if self.input_projection is not None:
            embedded = self.input_projection(embedded)
        return embedded

    def method_cell_for_depth(self, depth: int) -> SharedMethodCell:
        index = depth % self.config.method_cell_count
        return self.method_cell if index == 0 else self.additional_method_cells[index - 1]

    def output_logits(self, states: torch.Tensor) -> torch.Tensor:
        if self.output_embedding is None:
            logits = F.linear(states, self.token_embedding.weight)
        else:
            logits = F.linear(
                self.output_projection(states), self.output_embedding.weight
            )
        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias
        return logits

    def apply_depth_adapter(self, states: torch.Tensor, depth: int) -> torch.Tensor:
        if not self.depth_adapters:
            return states
        return states + self.depth_adapters[depth](states)

    @staticmethod
    def sinusoidal_positions(
        length: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / width)
        )
        encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: width // 2])
        return encoding.to(dtype=dtype)

    def forward(
        self, input_ids: torch.Tensor, return_method_states: bool = False
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        _, length = input_ids.shape
        if length > self.config.max_context:
            raise ValueError(f"sequence length {length} exceeds {self.config.max_context}")
        states = self.embed_tokens(input_ids) * math.sqrt(self.config.width)
        states = states + self.sinusoidal_positions(
            length, self.config.width, input_ids.device, states.dtype
        )[None, :, :]
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
        method_states = []
        for depth in range(self.config.repetitions):
            method_cell = self.method_cell_for_depth(depth)
            states = states + self.depth_embedding[depth]
            states = method_cell.local(states)
            if (depth + 1) % self.config.attention_every == 0:
                states = method_cell.global_attention(states, causal_mask)
            states = method_cell.feed_forward(states)
            states = self.apply_depth_adapter(states, depth)
            if return_method_states:
                method_states.append(states)
        states = self.final_norm(states)
        logits = self.output_logits(states)
        result: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {"logits": logits}
        if return_method_states:
            result["method_states"] = tuple(method_states)
        return result

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
