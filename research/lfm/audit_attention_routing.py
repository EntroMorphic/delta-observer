#!/usr/bin/env python3
"""Measure prompt separation and attention routing through every shared-cell stage."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from research.lfm.audit_observer_d import quantiles
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device


def find_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    return None


def separation(states: list[torch.Tensor]) -> dict:
    stacked = torch.stack(states).float()
    cosines = [
        float(F.cosine_similarity(stacked[first][None], stacked[second][None]).item())
        for first, second in itertools.combinations(range(len(states)), 2)
    ]
    distances = [
        float((stacked[first] - stacked[second]).norm().item())
        for first, second in itertools.combinations(range(len(states)), 2)
    ]
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square()
    fractions = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    return {
        "pairwise_cosine": quantiles(np.asarray(cosines)),
        "pairwise_euclidean_distance": quantiles(np.asarray(distances)),
        "state_norm": quantiles(stacked.norm(dim=-1).numpy()),
        "centered_energy": float(centered.square().sum(dim=-1).mean().item()),
        "mean_direction_energy": float(stacked.mean(dim=0).square().sum().item()),
        "mean_direction_energy_fraction": float(
            (
                stacked.mean(dim=0).square().sum()
                / stacked.square().sum(dim=-1).mean().clamp_min(1e-30)
            ).item()
        ),
        "effective_rank": float(
            torch.exp(-(fractions * fractions.clamp_min(1e-30).log()).sum()).item()
        ),
        "top_1_variance_fraction": float(fractions[0].item()),
    }


def route_prompt(
    model: ObserverDLM,
    input_ids: torch.Tensor,
    content_span: tuple[int, int] | None,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    length = input_ids.shape[1]
    states = model.embed_tokens(input_ids) * math.sqrt(model.config.width)
    states = states + model.sinusoidal_positions(
        length, model.config.width, input_ids.device, states.dtype
    )[None]
    causal_mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
    stage_states = {"embedding": states[0, -1].float().cpu()}
    routes = []
    for depth in range(model.config.repetitions):
        method_cell = model.method_cell_for_depth(depth)
        states = states + model.depth_embedding[depth]
        before_local = states
        states = method_cell.local(states)
        local_update = states - before_local
        stage_states[f"depth_{depth + 1}_local"] = states[0, -1].float().cpu()
        route = {
            "depth": depth + 1,
            "last_token_local_update_norm": float(local_update[0, -1].float().norm().item()),
        }
        if (depth + 1) % model.config.attention_every == 0:
            before_attention = states
            normalized = method_cell.attention_norm(states)
            attended, weights = method_cell.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=causal_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=True,
            )
            states = states + model.config.attention_residual_scale * attended
            last_weights = weights[0, :, -1].float()
            entropy = -(last_weights * last_weights.clamp_min(1e-30).log()).sum(dim=-1)
            route.update({
                "attention": True,
                "last_token_attention_update_norm": float(
                    (states - before_attention)[0, -1].float().norm().item()
                ),
                "head_entropy": entropy.cpu().tolist(),
                "head_effective_attended_tokens": entropy.exp().cpu().tolist(),
                "head_max_weight": last_weights.max(dim=-1).values.cpu().tolist(),
                "head_recent_8_mass": last_weights[:, max(0, length - 8) :].sum(dim=-1).cpu().tolist(),
                "head_first_token_mass": last_weights[:, 0].cpu().tolist(),
            })
            if content_span is not None:
                start, end = content_span
                route["content_span"] = [start, end]
                route["head_user_content_mass"] = (
                    last_weights[:, start:end].sum(dim=-1).cpu().tolist()
                )
            stage_states[f"depth_{depth + 1}_attention"] = states[0, -1].float().cpu()
        else:
            route["attention"] = False
        before_ffn = states
        states = method_cell.feed_forward(states)
        route["last_token_ffn_update_norm"] = float(
            (states - before_ffn)[0, -1].float().norm().item()
        )
        stage_states[f"depth_{depth + 1}_ffn"] = states[0, -1].float().cpu()
        before_adapter = states
        states = model.apply_depth_adapter(states, depth)
        route["last_token_adapter_update_norm"] = float(
            (states - before_adapter)[0, -1].float().norm().item()
        )
        stage_states[f"depth_{depth + 1}_adapter"] = states[0, -1].float().cpu()
        routes.append(route)
    return stage_states, routes


def audit_checkpoint(
    checkpoint_path: Path,
    prompts: list[dict],
    tokenizer,
    device: str,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ObserverDLM(ObserverDLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    all_stages: dict[str, list[torch.Tensor]] = {}
    prompt_routes = []
    with torch.inference_mode():
        for prompt in prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["text"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            ids = tokenizer(rendered, return_tensors="pt")["input_ids"]
            content_ids = tokenizer(
                prompt["text"], add_special_tokens=False
            )["input_ids"]
            span = find_subsequence(ids[0].tolist(), content_ids)
            stages, routes = route_prompt(model, ids.to(device), span)
            for name, state in stages.items():
                all_stages.setdefault(name, []).append(state)
            prompt_routes.append({
                "prompt_id": prompt["id"],
                "tokens": int(ids.shape[1]),
                "content_span": list(span) if span is not None else None,
                "routes": routes,
            })
    stage_separation = [
        {"stage": name, **separation(states)}
        for name, states in all_stages.items()
    ]
    attention_depths = []
    for depth in range(1, model.config.repetitions + 1):
        rows = [prompt["routes"][depth - 1] for prompt in prompt_routes]
        if not rows[0]["attention"]:
            continue
        attention_depths.append({
            "depth": depth,
            "user_content_mass": quantiles(np.asarray([
                value
                for row in rows
                for value in row.get("head_user_content_mass", [])
            ])),
            "recent_8_mass": quantiles(np.asarray([
                value for row in rows for value in row["head_recent_8_mass"]
            ])),
            "first_token_mass": quantiles(np.asarray([
                value for row in rows for value in row["head_first_token_mass"]
            ])),
            "attention_entropy": quantiles(np.asarray([
                value for row in rows for value in row["head_entropy"]
            ])),
            "effective_attended_tokens": quantiles(np.asarray([
                value for row in rows for value in row["head_effective_attended_tokens"]
            ])),
            "maximum_weight": quantiles(np.asarray([
                value for row in rows for value in row["head_max_weight"]
            ])),
            "attention_update_norm": quantiles(np.asarray([
                row["last_token_attention_update_norm"] for row in rows
            ])),
        })
    return {
        "checkpoint": str(checkpoint_path),
        "stage_prompt_separation": stage_separation,
        "attention_depths": attention_depths,
        "prompts": prompt_routes,
    }


def audit(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    tokenizer = AutoTokenizer.from_pretrained(arguments.teacher)
    prompts = json.loads(Path(arguments.prompts).read_text())["prompts"]
    result = {
        "device": device,
        "checkpoints": [
            audit_checkpoint(Path(path), prompts, tokenizer, device)
            for path in arguments.checkpoint
        ],
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--teacher", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--prompts", default="research/lfm/control_prompts.json")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_attention_routing_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps({
        checkpoint["checkpoint"]: checkpoint["attention_depths"]
        for checkpoint in result["checkpoints"]
    }, indent=2))
