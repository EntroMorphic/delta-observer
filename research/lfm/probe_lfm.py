#!/usr/bin/env python3
"""Probe LFM2.5 internal computational trajectories safely and compactly."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def random_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    return projection / output_dim**0.5


def probe(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, revision=arguments.revision)
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model, revision=arguments.revision, dtype=dtype
    )
    model.to(device).eval()

    encoded = tokenizer(arguments.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"][:, : arguments.max_tokens].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[:, : arguments.max_tokens].to(device)

    if device == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("LFM2.5 did not return hidden states")
    projection = random_projection(hidden_states[0].shape[-1], arguments.projection_dim, arguments.seed)
    projected = []
    delta_rms = []
    previous = None
    for hidden in hidden_states:
        current = hidden.detach().float().cpu()
        projected.append((current @ projection).to(torch.float16).numpy())
        if previous is not None:
            delta_rms.append(float((current - previous).square().mean().sqrt().item()))
        previous = current

    next_logits = outputs.logits[:, -1].float().cpu()
    top_values, top_indices = torch.topk(next_logits, arguments.top_k, dim=-1)
    probabilities = torch.softmax(next_logits, dim=-1)
    top_mass = probabilities.gather(-1, top_indices).sum(-1)
    layer_types = list(model.config.layer_types)
    result = {
        "model": arguments.model,
        "revision": arguments.revision,
        "prompt_sha256": hashlib.sha256(arguments.prompt.encode()).hexdigest(),
        "device": device,
        "dtype": str(dtype),
        "elapsed_seconds": elapsed,
        "input_tokens": int(input_ids.shape[1]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "layer_types": layer_types,
        "hidden_state_count": len(hidden_states),
        "hidden_state_shape": list(hidden_states[0].shape),
        "layer_delta_rms": delta_rms,
        "top_k_probability_mass": float(top_mass.item()),
        "top_tokens": tokenizer.convert_ids_to_tokens(top_indices[0].tolist()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }

    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination.with_suffix(".npz"),
        input_ids=input_ids.cpu().numpy(),
        projected_hidden_states=np.stack(projected),
        top_logit_indices=top_indices.numpy(),
        top_logit_values=top_values.numpy(),
        projection_seed=np.array(arguments.seed),
    )
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="research/results/lfm_probe")
    return parser.parse_args()


def main() -> None:
    result = probe(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
