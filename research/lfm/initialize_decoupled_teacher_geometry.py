#!/usr/bin/env python3
"""Initialize protected input geometry and a separate low-rank output head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from research.lfm.initialize_teacher_embedding import align_embedding_geometry
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig


def initialize(arguments: argparse.Namespace) -> dict:
    source = torch.load(arguments.base_checkpoint, map_location="cpu", weights_only=False)
    source_config = ObserverDLMConfig(**source["config"])
    if source_config.output_head_rank:
        raise ValueError("Base checkpoint must have the original tied output head")
    config = ObserverDLMConfig(**{
        **source_config.__dict__,
        "output_head_rank": arguments.output_head_rank,
        "attention_residual_scale": arguments.attention_residual_scale,
        "ffn_residual_scale": arguments.ffn_residual_scale,
    })
    model = ObserverDLM(config)
    incompatible = model.load_state_dict(source["model"], strict=False)
    expected_missing = {"output_projection.weight", "output_embedding.weight"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(f"Unexpected architecture mismatch: {incompatible}")

    source_weight = source["model"]["token_embedding.weight"].float()
    left, singular_values, right = torch.linalg.svd(source_weight, full_matrices=False)
    rank = arguments.output_head_rank
    root = singular_values[:rank].sqrt()
    with torch.no_grad():
        model.output_embedding.weight.copy_(left[:, :rank] * root)
        model.output_projection.weight.copy_(root[:, None] * right[:rank])

    teacher = AutoModelForCausalLM.from_pretrained(
        arguments.teacher, revision=arguments.revision, dtype=torch.float32
    ).eval()
    protected_input, geometry = align_embedding_geometry(
        teacher.get_input_embeddings().weight.detach(),
        source_weight,
        arguments.seed,
        arguments.blend,
    )
    with torch.no_grad():
        model.token_embedding.weight.copy_(protected_input)
    approximation = model.output_embedding.weight @ model.output_projection.weight
    result = {
        "method": "decoupled teacher input geometry plus truncated-SVD output head",
        "base_checkpoint": str(Path(arguments.base_checkpoint)),
        "teacher": arguments.teacher,
        "revision": arguments.revision,
        "blend": arguments.blend,
        "seed": arguments.seed,
        "output_head_rank": rank,
        "attention_residual_scale": arguments.attention_residual_scale,
        "ffn_residual_scale": arguments.ffn_residual_scale,
        "parameter_count": model.parameter_count(),
        "source_free_after_export": True,
        "output_weight_squared_energy_retained": float(
            (singular_values[:rank].square().sum() / singular_values.square().sum()).item()
        ),
        "output_weight_relative_frobenius_error": float(
            ((source_weight - approximation).norm() / source_weight.norm()).item()
        ),
        **geometry,
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": config.__dict__,
        "embedding_initialization": result,
    }, destination)
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--teacher", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--blend", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-head-rank", type=int, default=96)
    parser.add_argument("--attention-residual-scale", type=float, default=1.0)
    parser.add_argument("--ffn-residual-scale", type=float, default=0.75)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(initialize(parse_args()), indent=2))
