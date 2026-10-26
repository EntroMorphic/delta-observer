#!/usr/bin/env python3
"""Extract teacher corrections on Observer.D's own autoregressive histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.extract_lfm_corpus import hash_array, save_shard
from research.lfm.extract_lfm_instructions import normalize_text, prompt_stream
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device, random_projection


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(arguments: argparse.Namespace) -> dict:
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    device = choose_device(arguments.device)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        arguments.teacher, revision=arguments.teacher_revision
    )
    frozen_path = Path(arguments.frozen_prompts)
    frozen_bytes = frozen_path.read_bytes()
    frozen_registry = json.loads(frozen_bytes)
    excluded = {
        normalize_text(prompt["text"]) for prompt in frozen_registry["prompts"]
    }
    dataset = load_dataset(
        arguments.dataset,
        arguments.dataset_config,
        split=arguments.split,
        streaming=True,
        revision=arguments.dataset_revision,
    )
    prompt_stats = {
        "rejected_missing_user_text": 0,
        "rejected_frozen_prompt": 0,
        "rejected_duplicate_prompt": 0,
    }
    stream = prompt_stream(dataset, arguments.messages_field, excluded, prompt_stats)
    prior_hashes = set()
    if arguments.exclude_prompt_hashes:
        prior_hashes = set(Path(arguments.exclude_prompt_hashes).read_text().splitlines())
    accepted: list[tuple[str, list[int]]] = []
    rejected_too_long = 0
    rejected_prior_prompt = 0
    while len(accepted) < arguments.sequences:
        text = next(stream)
        prompt_hash = hashlib.sha256(normalize_text(text).encode()).hexdigest()
        if prompt_hash in prior_hashes:
            rejected_prior_prompt += 1
            continue
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = tokenizer(rendered, add_special_tokens=True)["input_ids"]
        if (
            len(prompt_ids) > arguments.max_prompt_tokens
            or len(prompt_ids) >= arguments.sequence_length
        ):
            rejected_too_long += 1
            continue
        accepted.append((text, prompt_ids))
    stream.close()

    student_path = Path(arguments.student_checkpoint)
    checkpoint = torch.load(student_path, map_location="cpu", weights_only=False)
    student = ObserverDLM(ObserverDLMConfig(**checkpoint["config"]))
    student.load_state_dict(checkpoint["model"])
    student.to(device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        arguments.teacher,
        revision=arguments.teacher_revision,
        dtype=dtype,
    ).to(device).eval()

    projection_array = random_projection(
        teacher.config.hidden_size, arguments.projection_dim, arguments.projection_seed
    ).numpy().astype(np.float32)
    np.save(destination / "projection.npy", projection_array)

    groups: dict[int, list[tuple[int, str, list[int]]]] = defaultdict(list)
    for index, (text, prompt_ids) in enumerate(accepted):
        groups[len(prompt_ids)].append((index, text, prompt_ids))
    completed: list[tuple[int, dict]] = []
    teacher_nll_sum = 0.0
    teacher_top_k_mass_sum = 0.0
    supervised_tokens = 0
    started = time.perf_counter()
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define EOS")

    for prompt_length in sorted(groups):
        group = groups[prompt_length]
        for start in range(0, len(group), arguments.batch_size):
            batch = group[start : start + arguments.batch_size]
            histories = torch.tensor(
                [item[2] for item in batch], dtype=torch.long, device=device
            )
            finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
            actual_lengths = torch.full(
                (len(batch),), arguments.sequence_length + 1, dtype=torch.long
            )
            with torch.inference_mode():
                while histories.shape[1] < arguments.sequence_length + 1:
                    next_tokens = student(histories)["logits"][:, -1].argmax(dim=-1)
                    next_tokens = torch.where(
                        finished, torch.full_like(next_tokens, eos), next_tokens
                    )
                    histories = torch.cat((histories, next_tokens[:, None]), dim=1)
                    newly_finished = (~finished) & (next_tokens == eos)
                    actual_lengths[newly_finished.cpu()] = histories.shape[1]
                    finished |= newly_finished
                input_ids = histories[:, :-1]
                outputs = teacher(input_ids=input_ids, use_cache=False)
                logits = outputs.logits.float()
                top_values, top_indices = torch.topk(logits, arguments.top_k, dim=-1)
                logsumexp = torch.logsumexp(logits, dim=-1)
            labels = top_indices[..., 0]
            teacher_label_logits = top_values[..., 0]
            loss_mask = torch.zeros_like(labels, dtype=torch.bool)
            for index, actual_length in enumerate(actual_lengths.tolist()):
                loss_mask[index, prompt_length - 1 : actual_length - 1] = True
            mask_float = loss_mask.float()
            supervised_tokens += int(loss_mask.sum().item())
            teacher_nll_sum += float(
                ((logsumexp - teacher_label_logits) * mask_float).sum().item()
            )
            teacher_top_k_mass_sum += float(
                (
                    torch.exp(top_values - logsumexp.unsqueeze(-1)).sum(dim=-1)
                    * mask_float
                ).sum().item()
            )
            for index, (source_index, _, _) in enumerate(batch):
                completed.append((source_index, {
                    "input_ids": input_ids[index].cpu().numpy(),
                    "labels": labels[index].cpu().numpy(),
                    "top_indices": top_indices[index].cpu().numpy(),
                    "top_values": top_values[index].cpu().numpy(),
                    "logsumexp": logsumexp[index].cpu().numpy(),
                    "teacher_label_logits": teacher_label_logits[index].cpu().numpy(),
                    "loss_mask": loss_mask[index].cpu().numpy(),
                }))
            if len(completed) % arguments.log_every < len(batch):
                print(f"Extracted {len(completed)}/{arguments.sequences}", flush=True)

    completed.sort(key=lambda item: item[0])
    rows = [row for _, row in completed]
    shards = []
    for shard_index, start in enumerate(range(0, len(rows), arguments.shard_size)):
        shards.append(save_shard(
            destination, shard_index, rows[start : start + arguments.shard_size]
        ))
    accepted_hashes = [
        hashlib.sha256(normalize_text(text).encode()).hexdigest()
        for text, _ in accepted
    ]
    prompt_hash_bytes = ("\n".join(accepted_hashes) + "\n").encode()
    (destination / "prompt_hashes.txt").write_bytes(prompt_hash_bytes)
    elapsed = time.perf_counter() - started
    manifest = {
        "format": "observer-d-lfm-on-policy-correction-v1",
        "config": {
            "teacher": arguments.teacher,
            "teacher_revision": arguments.teacher_revision,
            "student_checkpoint": str(student_path),
            "dataset": arguments.dataset,
            "dataset_revision": arguments.dataset_revision,
            "dataset_config": arguments.dataset_config,
            "split": arguments.split,
            "messages_field": arguments.messages_field,
            "sequences": arguments.sequences,
            "sequence_length": arguments.sequence_length,
            "max_prompt_tokens": arguments.max_prompt_tokens,
            "batch_size": arguments.batch_size,
            "shard_size": arguments.shard_size,
            "top_k": arguments.top_k,
            "projection_dim": arguments.projection_dim,
            "projection_seed": arguments.projection_seed,
            "target": "teacher top-1 and sparse distribution on student histories",
            "exclude_prompt_hashes": arguments.exclude_prompt_hashes,
        },
        "model_parameter_count": sum(p.numel() for p in teacher.parameters()),
        "student_parameter_count": student.parameter_count(),
        "student_checkpoint_sha256": file_sha256(student_path),
        "resolved_device": device,
        "dtype": str(dtype),
        "projection_shape": list(projection_array.shape),
        "projection_sha256": hash_array(projection_array),
        "prompt_hashes_path": "prompt_hashes.txt",
        "prompt_hashes_sha256": hashlib.sha256(prompt_hash_bytes).hexdigest(),
        "accepted_unique_prompts": len(accepted_hashes),
        "frozen_prompts_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "frozen_prompts_version": frozen_registry.get("version"),
        "excluded_frozen_prompt_count": len(excluded),
        **prompt_stats,
        "rejected_too_long": rejected_too_long,
        "rejected_prior_prompt": rejected_prior_prompt,
        "elapsed_seconds": elapsed,
        "tokens": arguments.sequences * arguments.sequence_length,
        "supervised_tokens": supervised_tokens,
        "teacher_top1_cross_entropy": teacher_nll_sum / supervised_tokens,
        "teacher_top_k_probability_mass": teacher_top_k_mass_sum / supervised_tokens,
        "shards": shards,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher", default="LiquidAI/LFM2.5-350M")
    parser.add_argument(
        "--teacher-revision", default="9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
    )
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument(
        "--dataset-revision", default="8049631c405ae6576f93f445c6b8166f76f5505a"
    )
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--sequences", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--frozen-prompts", default="research/lfm/control_prompts.json")
    parser.add_argument("--exclude-prompt-hashes")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
