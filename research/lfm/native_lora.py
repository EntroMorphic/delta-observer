"""Small explicit LoRA implementation for native Transformers linear operators."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive and finite")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=torch.float32
        ))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=torch.float32
        ))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        update = (inputs.to(self.lora_a.dtype) @ self.lora_a.T) @ self.lora_b.T
        return self.base(inputs) + (self.scale * update).to(inputs.dtype)


def install_lora(model: nn.Module, rank: int = 8, alpha: float = 16) -> list[str]:
    """Adapt native attention and short-convolution projections, not vocabulary/FFNs."""
    targets = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (".self_attn." in name or ".conv." in name)
        and name.rsplit(".", 1)[-1] in (
            "q_proj", "k_proj", "v_proj", "in_proj", "out_proj"
        )
    ]
    if not targets:
        raise ValueError("No native attention/conv linear targets found")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("Adapters already installed")
    model.requires_grad_(False)
    for name, base in targets:
        parent_name, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attribute, LoRALinear(base, rank, alpha))
    return [name for name, _ in targets]


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def load_adapter(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError("Adapter checkpoint keys do not match trainable parameters")
    for name, value in state.items():
        if value.shape != parameters[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter tensor: {name}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def supervised_chat(tokenizer, prompt: str, response: str, max_length: int) -> dict:
    """Teacher-forced causal row; no padding loss, duplicate BOS, or lost EOS."""
    if not prompt.strip() or not response.strip():
        raise ValueError("Prompt and response must be nonempty")
    if set(tokenizer.all_special_ids).intersection(
        tokenizer.encode(prompt, add_special_tokens=False)
    ):
        raise ValueError("Prompt contains tokenizer control tokens")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    answer = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires EOS")
    special_ids = set(tokenizer.all_special_ids)
    if special_ids.intersection(answer):
        raise ValueError("Response contains tokenizer control tokens")
    tokens = prefix + answer + [tokenizer.eos_token_id]
    if len(tokens) > max_length:
        raise ValueError("Example too long; truncation is forbidden")
    if not prefix or not answer:
        raise ValueError("Empty tokenized prompt or response")
    labels = [-100] * len(prefix) + answer + [tokenizer.eos_token_id]
    return {"input_ids": tokens, "labels": labels,
            "prompt_tokens": len(prefix), "supervised_tokens": len(answer) + 1}
