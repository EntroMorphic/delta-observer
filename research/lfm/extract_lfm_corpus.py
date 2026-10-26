#!/usr/bin/env python3
"""Extract sharded LFM2.5 behavior and computational trajectories at scale."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.probe_lfm import choose_device, random_projection


@dataclass(frozen=True)
class ExtractionConfig:
    model: str
    revision: str
    dataset: str
    dataset_config: str | None
    split: str
    text_field: str
    skip_sequences: int
    sequences: int
    sequence_length: int
    batch_size: int
    shard_size: int
    top_k: int
    projection_dim: int
    projection_seed: int
    capture_trajectories: bool
    log_every: int
    device: str
    output: str


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def token_sequences(dataset, tokenizer, text_field: str, length: int):
    """Pack streaming documents into fixed-length causal sequences."""
    buffer: list[int] = []
    eos = tokenizer.eos_token_id
    bos = tokenizer.bos_token_id
    if eos is None or bos is None:
        raise ValueError("Tokenizer must define BOS and EOS tokens")
    iterator = iter(dataset)
    try:
        for record in iterator:
            text = record.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            buffer.extend(tokenizer.encode(text, add_special_tokens=False))
            buffer.append(eos)
            while len(buffer) >= length:
                # Every independently evaluated chunk needs a valid recurrent
                # initial condition. LFM2.5 behaves pathologically without BOS.
                packed = [bos] + buffer[:length]
                del buffer[:length]
                yield packed
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def save_shard(destination: Path, shard_index: int, rows: list[dict]) -> dict:
    shard_path = destination / f"shard-{shard_index:05d}.npz"
    arrays = {
        "input_ids": np.stack([row["input_ids"] for row in rows]).astype(np.uint16),
        "labels": np.stack([row["labels"] for row in rows]).astype(np.uint16),
        "top_logit_indices": np.stack([row["top_indices"] for row in rows]).astype(np.uint16),
        "top_logit_values": np.stack([row["top_values"] for row in rows]).astype(np.float16),
        "teacher_logsumexp": np.stack([row["logsumexp"] for row in rows]).astype(np.float32),
        "teacher_label_logits": np.stack(
            [row["teacher_label_logits"] for row in rows]
        ).astype(np.float32),
    }
    if "projected" in rows[0]:
        arrays["projected_hidden_states"] = np.stack(
            [row["projected"] for row in rows]
        ).astype(np.float16)
    if "loss_mask" in rows[0]:
        arrays["loss_mask"] = np.stack([row["loss_mask"] for row in rows]).astype(np.uint8)
    np.savez(shard_path, **arrays)
    return {
        "path": shard_path.name,
        "sequences": len(rows),
        "bytes": shard_path.stat().st_size,
        "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
    }


def extract(config: ExtractionConfig) -> dict:
    destination = Path(config.output)
    destination.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision, dtype=dtype
    ).to(device).eval()
    dataset = load_dataset(
        config.dataset,
        config.dataset_config,
        split=config.split,
        streaming=True,
    )

    projection_cpu = random_projection(
        model.config.hidden_size, config.projection_dim, config.projection_seed
    )
    projection_array = projection_cpu.numpy().astype(np.float32)
    projection_path = destination / "projection.npy"
    np.save(projection_path, projection_array)
    projection = projection_cpu.to(device=device, dtype=dtype)

    stream = token_sequences(dataset, tokenizer, config.text_field, config.sequence_length)
    for _ in range(config.skip_sequences):
        next(stream)
    rows: list[dict] = []
    shards = []
    extracted = 0
    shard_index = 0
    started = time.perf_counter()
    teacher_nll_sum = 0.0
    teacher_top_k_mass_sum = 0.0
    observed_tokens = 0

    while extracted < config.sequences:
        batch_count = min(config.batch_size, config.sequences - extracted)
        packed = [next(stream) for _ in range(batch_count)]
        tokens = torch.tensor(packed, dtype=torch.long, device=device)
        input_ids, labels = tokens[:, :-1], tokens[:, 1:]
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=config.capture_trajectories,
                use_cache=False,
            )
            top_values, top_indices = torch.topk(outputs.logits.float(), config.top_k, dim=-1)
            logsumexp = torch.logsumexp(outputs.logits.float(), dim=-1)
            teacher_label_logits = outputs.logits.float().gather(
                -1, labels.unsqueeze(-1)
            ).squeeze(-1)
            teacher_nll_sum += float((logsumexp - teacher_label_logits).sum().item())
            teacher_top_k_mass_sum += float(
                torch.exp(top_values - logsumexp.unsqueeze(-1)).sum().item()
            )
            observed_tokens += labels.numel()
            projected = torch.stack(
                [hidden.to(dtype) @ projection for hidden in outputs.hidden_states], dim=1
            ) if config.capture_trajectories else None

        input_np = input_ids.cpu().numpy()
        labels_np = labels.cpu().numpy()
        indices_np = top_indices.cpu().numpy()
        values_np = top_values.cpu().numpy()
        logsumexp_np = logsumexp.cpu().numpy()
        teacher_label_logits_np = teacher_label_logits.cpu().numpy()
        projected_np = projected.cpu().numpy() if projected is not None else None
        for index in range(batch_count):
            rows.append({
                "input_ids": input_np[index],
                "labels": labels_np[index],
                "top_indices": indices_np[index],
                "top_values": values_np[index],
                "logsumexp": logsumexp_np[index],
                "teacher_label_logits": teacher_label_logits_np[index],
            })
            if projected_np is not None:
                rows[-1]["projected"] = projected_np[index]
        extracted += batch_count
        if extracted == config.sequences or extracted % config.log_every == 0:
            print(f"Extracted {extracted}/{config.sequences}", flush=True)

        if len(rows) >= config.shard_size or extracted == config.sequences:
            shards.append(save_shard(destination, shard_index, rows))
            rows.clear()
            shard_index += 1

    stream.close()

    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    manifest = {
        "format": "observer-d-lfm-trajectory-v1",
        "config": asdict(config),
        "resolved_device": device,
        "dtype": str(dtype),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "layer_types": list(model.config.layer_types),
        "hidden_state_count": model.config.num_hidden_layers + 1,
        "projection_shape": list(projection_array.shape),
        "projection_sha256": hash_array(projection_array),
        "elapsed_seconds": elapsed,
        "tokens": extracted * config.sequence_length,
        "tokens_per_second": extracted * config.sequence_length / elapsed,
        "teacher_cross_entropy": teacher_nll_sum / observed_tokens,
        "teacher_top_k_probability_mass": teacher_top_k_mass_sum / observed_tokens,
        "shards": shards,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> ExtractionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--skip-sequences", type=int, default=0)
    parser.add_argument("--sequences", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--no-trajectories", dest="capture_trajectories", action="store_false")
    parser.set_defaults(capture_trajectories=True)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_corpus")
    values = vars(parser.parse_args())
    if values["dataset_config"].lower() in ("none", "null", ""):
        values["dataset_config"] = None
    return ExtractionConfig(**values)


def main() -> None:
    extract(parse_args())
    # Some versions of the datasets streaming HTTP stack retain a non-daemon
    # prefetch worker after its iterator is closed. All shards and the manifest
    # are durable at this point; force a clean CLI boundary rather than hanging.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
