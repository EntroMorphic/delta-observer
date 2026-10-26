#!/usr/bin/env python3
"""Measure Observer.D failure modes at token, position, depth, and parameter level."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from research.lfm.distillation import sparse_topk_kl
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device
from research.lfm.train_observer_d_lm import TrajectoryShards


def quantiles(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def forward_at_depth(model: ObserverDLM, input_ids: torch.Tensor, depth_count: int) -> dict:
    _, length = input_ids.shape
    states = model.embed_tokens(input_ids) * math.sqrt(model.config.width)
    states = states + model.sinusoidal_positions(
        length, model.config.width, input_ids.device, states.dtype
    )[None]
    causal_mask = torch.ones(
        length, length, dtype=torch.bool, device=input_ids.device
    ).triu(1)
    method_states = []
    for depth in range(depth_count):
        method_cell = model.method_cell_for_depth(depth)
        states = states + model.depth_embedding[depth]
        states = method_cell.local(states)
        if (depth + 1) % model.config.attention_every == 0:
            states = method_cell.global_attention(states, causal_mask)
        states = method_cell.feed_forward(states)
        states = model.apply_depth_adapter(states, depth)
        method_states.append(states)
    logits = model.output_logits(model.final_norm(states))
    return {"logits": logits, "method_states": tuple(method_states)}


def parameter_bytes(model: torch.nn.Module) -> dict:
    groups = defaultdict(lambda: {"parameters": 0, "bytes": 0})
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        groups[group]["parameters"] += parameter.numel()
        groups[group]["bytes"] += parameter.numel() * parameter.element_size()
    return {
        "total_parameters": sum(value["parameters"] for value in groups.values()),
        "total_resident_bytes": sum(value["bytes"] for value in groups.values()),
        "by_top_level_module": dict(groups),
        "persistent_kv_cache_bytes": 0,
        "persistent_kv_cache_reason": "Observer.D forward does not implement a KV cache",
    }


def embedding_geometry(model: ObserverDLM) -> dict:
    weight = model.token_embedding.weight.detach().float().cpu()
    centered = weight - weight.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    normalized = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    effective_rank = torch.exp(-(normalized * normalized.clamp_min(1e-30).log()).sum())
    row_norms = weight.norm(dim=-1).numpy()
    return {
        "shape": list(weight.shape),
        "mean": float(weight.mean().item()),
        "std": float(weight.std().item()),
        "row_norm": quantiles(row_norms),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "effective_rank": float(effective_rank.item()),
        "stable_rank": float((eigenvalues.sum() / eigenvalues[0].clamp_min(1e-30)).item()),
        "top_1_variance_fraction": float(normalized[0].item()),
        "top_16_variance_fraction": float(normalized[:16].sum().item()),
        "top_64_variance_fraction": float(normalized[:64].sum().item()),
        "zero_norm_rows": int((torch.from_numpy(row_norms) == 0).sum().item()),
    }


def verify_corpus(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    checks = []
    for shard in manifest["shards"]:
        path = directory / shard["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append({
            "path": shard["path"],
            "bytes": path.stat().st_size,
            "expected_sha256": shard["sha256"],
            "actual_sha256": actual,
            "match": actual == shard["sha256"],
        })
    return {
        "manifest": str(directory / "manifest.json"),
        "format": manifest["format"],
        "tokens": manifest["tokens"],
        "shards": len(checks),
        "total_shard_bytes": sum(check["bytes"] for check in checks),
        "all_hashes_match": all(check["match"] for check in checks),
        "checks": checks,
    }


def exposure_counts(directories: list[Path]) -> Counter:
    counts = Counter()
    for directory in directories:
        for shard in sorted(directory.glob("shard-*.npz")):
            with np.load(shard) as data:
                labels = data["labels"].astype(np.int64)
                if "loss_mask" in data:
                    labels = labels[data["loss_mask"].astype(bool)]
                counts.update(labels.reshape(-1).tolist())
    return counts


def gradient_measurement(
    model: ObserverDLM, batch: dict, device: str, mechanism: str
) -> dict:
    model.zero_grad(set_to_none=True)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["loss_mask"].to(device)
    logits = model(input_ids)["logits"]
    token_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    ce = (token_ce * mask).sum() / mask.sum()
    kl = torch.zeros((), device=device)
    if mechanism == "distill":
        kl = sparse_topk_kl(
            logits,
            batch["top_indices"].to(device),
            batch["top_values"].to(device),
            batch["logsumexp"].to(device),
            mask=mask,
        )
    (ce + kl).backward()
    groups = defaultdict(lambda: {"squared_norm": 0.0, "parameters": 0, "nonzero": 0})
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        groups[group]["parameters"] += parameter.numel()
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            groups[group]["squared_norm"] += float(gradient.square().sum().item())
            groups[group]["nonzero"] += int(torch.count_nonzero(gradient).item())
    result = {}
    for name, values in groups.items():
        result[name] = {
            "l2_norm": math.sqrt(values.pop("squared_norm")),
            **values,
        }
    model.zero_grad(set_to_none=True)
    return {"ce": float(ce.item()), "sparse_kl": float(kl.item()), "groups": result}


def audit(arguments: argparse.Namespace) -> dict:
    device = choose_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    model = ObserverDLM(ObserverDLMConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(arguments.teacher)
    validation_directory = Path(arguments.validation_data)
    shards = sorted(validation_directory.glob("shard-*.npz"))
    dataset = TrajectoryShards(validation_directory, shuffle=False, shards=shards)
    loader = DataLoader(dataset, batch_size=arguments.batch_size, num_workers=0)
    exposure = exposure_counts([Path(path) for path in arguments.training_data])

    ce_values = []
    teacher_ce_values = []
    gap_values = []
    entropy_values = []
    confidence_values = []
    teacher_top1_probability_values = []
    student_teacher_top1_probability_values = []
    top_k_mass_values = []
    position_ce_sum = np.zeros(model.config.max_context, dtype=np.float64)
    position_teacher_ce_sum = np.zeros(model.config.max_context, dtype=np.float64)
    position_count = np.zeros(model.config.max_context, dtype=np.int64)
    token_loss_sum = defaultdict(float)
    token_loss_count = Counter()
    frequency_bucket_losses = defaultdict(list)
    depth_loss_sum = np.zeros(model.config.repetitions, dtype=np.float64)
    depth_state_norm_sum = np.zeros(model.config.repetitions, dtype=np.float64)
    depth_update_norm_sum = np.zeros(model.config.repetitions, dtype=np.float64)
    depth_batches = 0
    audited_sequences = 0
    first_batch = None
    with torch.inference_mode():
        for batch in loader:
            if audited_sequences >= arguments.sequences:
                break
            remaining = arguments.sequences - audited_sequences
            take = min(remaining, batch["input_ids"].shape[0])
            batch = {key: value[:take] for key, value in batch.items()}
            if first_batch is None:
                first_batch = {key: value.clone() for key, value in batch.items()}
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["loss_mask"].to(device).bool()
            output = model(input_ids, return_method_states=True)
            logits = output["logits"].float()
            log_probs = F.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            token_ce = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            teacher_ce = (
                batch["logsumexp"].to(device) - batch["teacher_label_logits"].to(device)
            )
            entropy = -(probabilities * log_probs).sum(dim=-1)
            confidence = probabilities.max(dim=-1).values
            teacher_top1 = batch["top_indices"][..., 0].to(device)
            student_teacher_top1 = probabilities.gather(
                -1, teacher_top1.unsqueeze(-1)
            ).squeeze(-1)
            teacher_top1_probability = torch.exp(
                batch["top_values"][..., 0].to(device) - batch["logsumexp"].to(device)
            )
            top_k_mass = torch.exp(
                batch["top_values"].to(device)
                - batch["logsumexp"].to(device).unsqueeze(-1)
            ).sum(dim=-1)

            for target, source in (
                (ce_values, token_ce),
                (teacher_ce_values, teacher_ce),
                (gap_values, token_ce - teacher_ce),
                (entropy_values, entropy),
                (confidence_values, confidence),
                (teacher_top1_probability_values, teacher_top1_probability),
                (student_teacher_top1_probability_values, student_teacher_top1),
                (top_k_mass_values, top_k_mass),
            ):
                target.extend(source[mask].cpu().tolist())
            mask_cpu = mask.cpu().numpy()
            ce_cpu = token_ce.cpu().numpy()
            teacher_ce_cpu = teacher_ce.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            for position in range(labels.shape[1]):
                selected = mask_cpu[:, position]
                position_ce_sum[position] += ce_cpu[selected, position].sum()
                position_teacher_ce_sum[position] += teacher_ce_cpu[selected, position].sum()
                position_count[position] += selected.sum()
            for token, loss, selected in zip(
                labels_cpu.reshape(-1), ce_cpu.reshape(-1), mask_cpu.reshape(-1)
            ):
                if selected:
                    token_loss_sum[int(token)] += float(loss)
                    token_loss_count[int(token)] += 1
                    count = exposure[int(token)]
                    bucket = "0" if count == 0 else "1-3" if count <= 3 else "4-15" if count <= 15 else "16-63" if count <= 63 else "64+"
                    frequency_bucket_losses[bucket].append(float(loss))

            previous = None
            for depth, states in enumerate(output["method_states"]):
                depth_state_norm_sum[depth] += float(states.float().norm(dim=-1).mean().item())
                if previous is not None:
                    depth_update_norm_sum[depth] += float(
                        (states.float() - previous.float()).norm(dim=-1).mean().item()
                    )
                previous = states
            for depth in range(1, model.config.repetitions + 1):
                depth_logits = forward_at_depth(model, input_ids, depth)["logits"].float()
                losses = F.cross_entropy(
                    depth_logits.reshape(-1, depth_logits.shape[-1]),
                    labels.reshape(-1), reduction="none"
                ).reshape_as(labels)
                depth_loss_sum[depth - 1] += float(losses[mask].mean().item())
            depth_batches += 1
            audited_sequences += take

    valid_positions = position_count > 0
    hard_tokens = sorted(
        (
            {
                "token_id": token,
                "token": tokenizer.convert_ids_to_tokens(token),
                "decoded": tokenizer.decode([token]),
                "validation_count": token_loss_count[token],
                "training_exposure": exposure[token],
                "mean_ce": token_loss_sum[token] / token_loss_count[token],
            }
            for token in token_loss_count
            if token_loss_count[token] >= arguments.minimum_token_count
        ),
        key=lambda row: row["mean_ce"], reverse=True
    )[: arguments.hard_tokens]

    gradient = gradient_measurement(model, first_batch, device, "distill")
    timing_input = first_batch["input_ids"].to(device)
    with torch.inference_mode():
        for _ in range(arguments.warmup):
            model(timing_input)
        if device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(arguments.timing_steps):
            model(timing_input)
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started

    result = {
        "checkpoint": str(Path(arguments.checkpoint)),
        "device": device,
        "audit_sequences": audited_sequences,
        "audit_supervised_tokens": len(ce_values),
        "corpus_integrity": verify_corpus(validation_directory),
        "parameter_storage": parameter_bytes(model),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "embedding_geometry": embedding_geometry(model),
        "token_metrics": {
            "student_ce": quantiles(np.asarray(ce_values)),
            "teacher_ce": quantiles(np.asarray(teacher_ce_values)),
            "student_minus_teacher_ce": quantiles(np.asarray(gap_values)),
            "student_entropy": quantiles(np.asarray(entropy_values)),
            "student_top1_confidence": quantiles(np.asarray(confidence_values)),
            "teacher_top1_probability": quantiles(np.asarray(teacher_top1_probability_values)),
            "student_probability_on_teacher_top1": quantiles(np.asarray(student_teacher_top1_probability_values)),
            "teacher_top_k_mass": quantiles(np.asarray(top_k_mass_values)),
        },
        "position_metrics": {
            "count": position_count[valid_positions].tolist(),
            "student_ce": (position_ce_sum[valid_positions] / position_count[valid_positions]).tolist(),
            "teacher_ce": (position_teacher_ce_sum[valid_positions] / position_count[valid_positions]).tolist(),
        },
        "training_frequency_buckets": {
            bucket: quantiles(np.asarray(losses))
            for bucket, losses in frequency_bucket_losses.items()
        },
        "hard_tokens": hard_tokens,
        "depth_metrics": {
            "ce": (depth_loss_sum / depth_batches).tolist(),
            "state_norm": (depth_state_norm_sum / depth_batches).tolist(),
            "update_norm": (depth_update_norm_sum / depth_batches).tolist(),
        },
        "gradient_measurement": gradient,
        "native_timing": {
            "batch_size": int(timing_input.shape[0]),
            "sequence_length": int(timing_input.shape[1]),
            "steps": arguments.timing_steps,
            "elapsed_seconds": elapsed,
            "tokens_per_second": timing_input.numel() * arguments.timing_steps / elapsed,
        },
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--training-data", nargs="+", required=True)
    parser.add_argument("--teacher", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--minimum-token-count", type=int, default=2)
    parser.add_argument("--hard-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing-steps", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/observer_d_atomic_audit.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    summary = {
        "output": result["checkpoint"],
        "audit_sequences": result["audit_sequences"],
        "student_ce": result["token_metrics"]["student_ce"]["mean"],
        "teacher_ce": result["token_metrics"]["teacher_ce"]["mean"],
        "effective_embedding_rank": result["embedding_geometry"]["effective_rank"],
        "tokens_per_second": result["native_timing"]["tokens_per_second"],
    }
    print(json.dumps(summary, indent=2))
