#!/usr/bin/env python3
"""Train budget-matched native LFM adapters with auditable deterministic resume."""

import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.evaluate_pretrained_development import synchronize
from research.lfm.native_lora import adapter_state, install_lora, load_adapter, supervised_chat
from research.lfm.native_lora_smoke import MODEL, REVISION, frozen_digest
from research.lfm.probe_lfm import choose_device


def run(args):
    directory = Path(args.data)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    spec = manifest["arms"][args.arm]
    data_bytes = (directory / spec["path"]).read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != spec["sha256"]:
        raise ValueError("Training data hash mismatch")
    if manifest["model"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("Unexpected base model")
    if args.epochs < 1 or args.rank < 1:
        raise ValueError("Epoch and rank budgets must be positive")
    if args.snapshot_every < 0:
        raise ValueError("Snapshot interval cannot be negative")
    if args.max_steps < 1 or args.max_steps > spec["examples"] * args.epochs:
        raise ValueError("Steps must be within the fixed epoch budget")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    rows = json.loads(data_bytes)
    if (len(rows) != spec["examples"]
            or sum(row["supervised_tokens"] for row in rows) != spec["supervised_tokens"]
            or sum(len(row["input_ids"]) for row in rows) != spec["input_tokens"]):
        raise ValueError("Manifest example/token budget differs from data")
    for row in rows:
        encoded = supervised_chat(tokenizer, row["prompt"], row["response"], 128)
        if any(row[k] != value for k, value in encoded.items()):
            raise ValueError("Serialized tokenization or loss mask is invalid")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32
    ).to(device).eval()
    targets = install_lora(model, args.rank, 2 * args.rank)
    frozen_before = frozen_digest(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01
    )
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    if "training_order" in spec:
        order = spec["training_order"]
        if (not isinstance(order, list) or any(type(i) is not int for i in order)
                or sorted(order) != list(range(len(rows)))):
            raise ValueError("Training order must be a permutation of row indices")
    config = {"model": MODEL, "revision": REVISION, "seed": args.seed,
        "arm": args.arm, "rank": args.rank, "alpha": 2 * args.rank, "targets": targets,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "data_sha256": spec["sha256"], "learning_rate": 1e-4,
        "weight_decay": .01, "gradient_clip": 1., "batch_size": 1,
        "dtype": "float32", "max_steps": args.max_steps,
        "epochs": args.epochs, "epoch_order_policy": "repeat_fixed_permutation",
        "snapshot_every": args.snapshot_every}
    config["training_order_sha256"] = hashlib.sha256(
        json.dumps(order).encode()
    ).hexdigest()
    history = []
    if args.resume:
        saved = torch.load(args.resume, weights_only=True, map_location="cpu")
        # Budget may be extended after an explicitly labeled engineering run.
        for key, value in config.items():
            if key not in ("max_steps", "epochs", "snapshot_every") and saved["config"].get(key) != value:
                raise ValueError(f"Resume configuration differs: {key}")
        for index, record in enumerate(saved['history']):
            if (record['step'] != index + 1 or record['row_index'] != order[index % len(order)]):
                raise ValueError("Resume exposure history does not match fixed data order")
        load_adapter(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        if len(history) > args.max_steps:
            raise ValueError("Resume is beyond target budget")
    synchronize(device)
    start = time.perf_counter()
    for step in range(len(history), args.max_steps):
        row_index = order[step % len(order)]
        row = rows[row_index]
        inputs = torch.tensor([row["input_ids"]], device=device)
        labels = torch.tensor([row["labels"]], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True
        )
        optimizer.step()
        history.append({"step": step + 1, "row_index": row_index,
            "epoch": step // len(order),
            "loss": float(loss.detach()), "gradient_norm": float(norm),
            "input_tokens": len(row["input_ids"]),
            "supervised_tokens": row["supervised_tokens"]})
        checkpoint = {"config": config, "adapter": adapter_state(model),
            "optimizer": optimizer.state_dict(), "history": history}
        temporary = output / "checkpoint.pending.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "checkpoint.pt")
        if args.snapshot_every and (step + 1) % args.snapshot_every == 0:
            snapshot = output / f"checkpoint-step-{step + 1:06d}.pt"
            with snapshot.open('xb') as destination:
                torch.save(checkpoint, destination)
        print(json.dumps(history[-1]), flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    frozen_after = frozen_digest(model)
    report = {"config": config, "history": history,
        "steps": len(history), "new_update_seconds": elapsed,
        "supervised_tokens": sum(r["supervised_tokens"] for r in history),
        "input_tokens": sum(r["input_tokens"] for r in history),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_before_sha256": frozen_before, "frozen_after_sha256": frozen_after,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if device == "mps" else None,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_memory_bytes": None,
        "memory_note": "macOS ru_maxrss is peak process RSS in bytes; not peak Metal/unified-memory usage",
        "scope": "exploratory native adaptation, not confirmatory capability evidence"}
    (output / "training.json").write_text(json.dumps(report, indent=2))
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen base changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--arm", choices=("unique", "repeated", "remediation"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    run(parser.parse_args())
