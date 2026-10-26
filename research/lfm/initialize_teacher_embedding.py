#!/usr/bin/env python3
"""Compress LFM token geometry into a source-free Observer.D checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import random_projection


def align_embedding_geometry(
    teacher_embedding: torch.Tensor,
    student_embedding: torch.Tensor,
    seed: int,
    blend: float,
) -> tuple[torch.Tensor, dict]:
    """JL-project teacher tokens, Procrustes-align them, and blend with a student."""
    if teacher_embedding.shape[0] != student_embedding.shape[0]:
        raise ValueError("Teacher and student vocabularies differ")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    projection = random_projection(
        teacher_embedding.shape[1], student_embedding.shape[1], seed
    )
    projected = teacher_embedding.float().cpu() @ projection
    student = student_embedding.float().cpu()
    projected_mean = projected.mean(dim=0, keepdim=True)
    student_mean = student.mean(dim=0, keepdim=True)
    projected_centered = projected - projected_mean
    student_centered = student - student_mean
    left, _, right_t = torch.linalg.svd(projected_centered.T @ student_centered)
    rotation = left @ right_t
    aligned = projected_centered @ rotation
    aligned = aligned * (student_centered.std() / aligned.std().clamp_min(1e-12))
    aligned = aligned + student_mean
    combined = torch.lerp(student, aligned, blend)
    metrics = {
        "projection_shape": list(projection.shape),
        "projection_sha256": hashlib.sha256(projection.numpy().tobytes()).hexdigest(),
        "teacher_embedding_std": float(teacher_embedding.float().std().item()),
        "student_embedding_std_before": float(student.std().item()),
        "aligned_embedding_std": float(aligned.std().item()),
        "student_embedding_std_after": float(combined.std().item()),
        "aligned_student_cosine": float(torch.nn.functional.cosine_similarity(
            aligned, student, dim=-1
        ).mean().item()),
    }
    return combined, metrics


def initialize(arguments: argparse.Namespace) -> dict:
    base = torch.load(arguments.base_checkpoint, map_location="cpu", weights_only=False)
    config = ObserverDLMConfig(**base["config"])
    observer = ObserverDLM(config)
    observer.load_state_dict(base["model"])
    teacher = AutoModelForCausalLM.from_pretrained(
        arguments.teacher, revision=arguments.revision, dtype=torch.float32
    ).eval()
    teacher_embedding = teacher.get_input_embeddings().weight.detach()
    if teacher_embedding.shape[0] != config.vocab_size:
        raise ValueError("Teacher vocabulary does not match Observer.D")
    combined, metrics = align_embedding_geometry(
        teacher_embedding,
        observer.token_embedding.weight.detach(),
        arguments.seed,
        arguments.blend,
    )
    with torch.no_grad():
        observer.token_embedding.weight.copy_(combined)

    result = {
        "method": "JL projection plus orthogonal Procrustes alignment",
        "teacher": arguments.teacher,
        "revision": arguments.revision,
        "base_checkpoint": str(Path(arguments.base_checkpoint)),
        "blend": arguments.blend,
        "seed": arguments.seed,
        "parameter_count": observer.parameter_count(),
        "source_free_after_export": True,
        **metrics,
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": observer.state_dict(),
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
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="research/results/observer-d-teacher-embedding.pt")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(initialize(parse_args()), indent=2))
