#!/usr/bin/env python3
"""Train Observer.D-LM with direct, output, or trajectory supervision."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from research.lfm.distillation import (
    TrajectoryAlignment,
    conditional_assignment_loss,
    conditional_contrast_loss,
    masked_mean,
    sparse_topk_kl,
)
from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig
from research.lfm.probe_lfm import choose_device


class TrajectoryShards(IterableDataset):
    """Stream rows one shard at a time to keep memory bounded."""

    def __init__(
        self,
        directory: Path,
        seed: int = 0,
        shuffle: bool = True,
        shards: list[Path] | None = None,
    ):
        super().__init__()
        self.directory = directory
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.shards = sorted(directory.glob("shard-*.npz")) if shards is None else shards
        if not self.shards:
            raise FileNotFoundError(f"No trajectory shards found in {directory}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shards = list(self.shards)
        if self.shuffle:
            rng.shuffle(shards)
        for path in shards:
            with np.load(path) as data:
                count = len(data["input_ids"])
                order = list(range(count))
                if self.shuffle:
                    rng.shuffle(order)
                for index in order:
                    row = {
                        "input_ids": torch.from_numpy(data["input_ids"][index].astype(np.int64)),
                        "labels": torch.from_numpy(data["labels"][index].astype(np.int64)),
                        "top_indices": torch.from_numpy(
                            data["top_logit_indices"][index].astype(np.int64)
                        ),
                        "top_values": torch.from_numpy(
                            data["top_logit_values"][index].astype(np.float32)
                        ),
                        "logsumexp": torch.from_numpy(
                            data["teacher_logsumexp"][index].astype(np.float32)
                        ),
                        **({
                            "teacher_label_logits": torch.from_numpy(
                                data["teacher_label_logits"][index].astype(np.float32)
                            )
                        } if "teacher_label_logits" in data else {}),
                    }
                    if "projected_hidden_states" in data:
                        row["trajectory"] = torch.from_numpy(
                            data["projected_hidden_states"][index].astype(np.float32)
                        )
                    row["loss_mask"] = torch.from_numpy(
                        data["loss_mask"][index].astype(np.float32)
                    ) if "loss_mask" in data else torch.ones_like(
                        row["labels"], dtype=torch.float32
                    )
                    yield row


def evaluate(model, alignment, loader, mechanism, device) -> dict:
    totals = {"ce": 0.0, "distillation": 0.0, "state": 0.0, "delta": 0.0}
    teacher_ce = 0.0
    teacher_ce_tokens = 0.0
    teacher_agreement = 0.0
    supervised_tokens = 0.0
    batches = 0
    model.eval()
    alignment.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device), return_method_states=mechanism == "trajectory"
            )
            logits = output["logits"]
            labels = batch["labels"].to(device)
            mask = batch["loss_mask"].to(device)
            token_count = float(mask.sum().item())
            ce = masked_mean(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels), mask)
            totals["ce"] += float(ce.item()) * token_count
            teacher_top1 = batch["top_indices"][..., 0].to(device)
            agreement = masked_mean((logits.argmax(dim=-1) == teacher_top1).float(), mask)
            teacher_agreement += float(agreement.item()) * token_count
            supervised_tokens += token_count
            if "teacher_label_logits" in batch:
                batch_teacher_ce = masked_mean((
                    batch["logsumexp"].to(device)
                    - batch["teacher_label_logits"].to(device)
                ), mask)
                teacher_ce += float(batch_teacher_ce.item()) * token_count
                teacher_ce_tokens += token_count
            if mechanism in ("distill", "trajectory"):
                distillation = sparse_topk_kl(
                    logits,
                    batch["top_indices"].to(device),
                    batch["top_values"].to(device),
                    batch["logsumexp"].to(device),
                    mask=mask,
                )
                totals["distillation"] += float(distillation.item()) * token_count
            if mechanism == "trajectory":
                trajectory = alignment(
                    output["method_states"], batch["trajectory"].to(device), mask=mask
                )
                totals["state"] += float(trajectory["state"].item()) * token_count
                totals["delta"] += float(trajectory["delta"].item()) * token_count
            batches += 1
    model.train()
    alignment.train()
    result = {name: value / max(1.0, supervised_tokens) for name, value in totals.items()}
    result.update({
        "teacher_ce": teacher_ce / teacher_ce_tokens if teacher_ce_tokens else None,
        "teacher_top1_agreement": teacher_agreement / max(1.0, supervised_tokens),
        "supervised_tokens": int(supervised_tokens),
        "batches": batches,
    })
    return result


def train(arguments: argparse.Namespace) -> dict:
    if arguments.resume and arguments.initialize_from:
        raise ValueError("Use either --resume or --initialize-from, not both")
    torch.manual_seed(arguments.seed)
    random.seed(arguments.seed)
    directory = Path(arguments.data)
    manifest = json.loads((directory / "manifest.json").read_text())
    training_directories = [directory] + [Path(path) for path in arguments.additional_data]
    training_manifests = [manifest]
    for additional_directory in training_directories[1:]:
        additional_manifest = json.loads((additional_directory / "manifest.json").read_text())
        if manifest["projection_sha256"] != additional_manifest["projection_sha256"]:
            raise ValueError("Training projection matrices differ")
        if manifest["config"]["sequence_length"] != additional_manifest["config"]["sequence_length"]:
            raise ValueError("Training sequence lengths differ")
        training_manifests.append(additional_manifest)
    paired_directory = Path(arguments.paired_data) if arguments.paired_data else None
    if paired_directory is not None:
        if arguments.mechanism != "distill":
            raise ValueError("Paired correction currently requires distillation")
        paired_manifest = json.loads((paired_directory / "manifest.json").read_text())
        if manifest["projection_sha256"] != paired_manifest["projection_sha256"]:
            raise ValueError("Paired correction projection matrix differs")
        if manifest["config"]["sequence_length"] != paired_manifest["config"]["sequence_length"]:
            raise ValueError("Paired correction sequence length differs")
    config = ObserverDLMConfig(
        vocab_size=manifest["config"].get("vocab_size", 65_536),
        width=arguments.width,
        repetitions=arguments.repetitions,
        attention_every=arguments.attention_every,
        attention_heads=arguments.attention_heads,
        expansion=arguments.expansion,
        max_context=manifest["config"]["sequence_length"],
        output_head_rank=arguments.output_head_rank,
        output_bias=arguments.output_bias,
        attention_residual_scale=arguments.attention_residual_scale,
        ffn_residual_scale=arguments.ffn_residual_scale,
        depth_adapter_rank=arguments.depth_adapter_rank,
        input_embedding_rank=arguments.input_embedding_rank,
        method_cell_count=arguments.method_cell_count,
    )
    device = choose_device(arguments.device)
    model = ObserverDLM(config).to(device)
    projection_dim = manifest["projection_shape"][1]
    alignment = TrajectoryAlignment(config.width, projection_dim).to(device)
    if arguments.freeze_output_table:
        if model.output_embedding is None:
            raise ValueError("--freeze-output-table requires a separate output head")
        model.output_embedding.weight.requires_grad_(False)
    if arguments.freeze_output_projection:
        if model.output_projection is None:
            raise ValueError("--freeze-output-projection requires a separate output head")
        model.output_projection.weight.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if arguments.mechanism == "trajectory":
        parameters.extend(alignment.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    start_epoch = 0
    resume_batch = -1
    step = 0
    history = []
    if arguments.resume:
        checkpoint = torch.load(arguments.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        alignment.load_state_dict(checkpoint["alignment"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        resume_batch = checkpoint["batch_index"]
        step = checkpoint["step"]
        history = checkpoint.get("history", [])
    elif arguments.initialize_from:
        checkpoint = torch.load(arguments.initialize_from, map_location="cpu", weights_only=False)
        source_config = ObserverDLMConfig(**checkpoint["config"])
        shared_fields = (
            "vocab_size", "width", "repetitions", "attention_every",
            "attention_heads", "expansion", "local_kernel", "max_context",
        )
        if any(getattr(source_config, field) != getattr(config, field) for field in shared_fields):
            raise ValueError("Initialization checkpoint core architecture differs")
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = {"lm_head_bias", "output_projection.weight", "output_embedding.weight"}
        allowed_missing.update(
            key for key in incompatible.missing_keys if key.startswith("depth_adapters.")
        )
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                f"Unsupported initialization mismatch: {incompatible}"
            )
        if config.output_head_rank and not source_config.output_head_rank:
            source_weight = checkpoint["model"]["token_embedding.weight"].float()
            _, singular_values, right = torch.linalg.svd(
                source_weight, full_matrices=False
            )
            rank = config.output_head_rank
            root = singular_values[:rank].sqrt()
            # U*S is obtained without retaining U: W @ V / sqrt(S).
            output_embedding = (source_weight @ right[:rank].T) / root
            output_projection = root[:, None] * right[:rank]
            with torch.no_grad():
                model.output_embedding.weight.copy_(output_embedding.to(device))
                model.output_projection.weight.copy_(output_projection.to(device))
        if arguments.mechanism == "trajectory" and "alignment" in checkpoint:
            alignment.load_state_dict(checkpoint["alignment"])
    all_shards = [
        shard
        for training_directory in training_directories
        for shard in sorted(training_directory.glob("shard-*.npz"))
    ]
    if arguments.mechanism == "trajectory":
        for shard in all_shards:
            with np.load(shard) as data:
                if "projected_hidden_states" not in data:
                    raise ValueError(f"Trajectory mechanism requires hidden states: {shard}")
    if arguments.validation_data:
        validation_directory = Path(arguments.validation_data)
        validation_manifest = json.loads((validation_directory / "manifest.json").read_text())
        if manifest["projection_sha256"] != validation_manifest["projection_sha256"]:
            raise ValueError("Training and validation projection matrices differ")
        if manifest["config"]["sequence_length"] != validation_manifest["config"]["sequence_length"]:
            raise ValueError("Training and validation sequence lengths differ")
        train_shards = all_shards
        validation_shards = sorted(validation_directory.glob("shard-*.npz"))
    else:
        validation_directory = directory
        if arguments.holdout_shards < 1 or len(all_shards) <= arguments.holdout_shards:
            raise ValueError("holdout-shards must leave at least one training shard")
        train_shards = all_shards[:-arguments.holdout_shards]
        validation_shards = all_shards[-arguments.holdout_shards:]
    dataset = TrajectoryShards(directory, arguments.seed, shards=train_shards)
    validation_dataset = TrajectoryShards(
        validation_directory, arguments.seed, shuffle=False, shards=validation_shards
    )
    loader = DataLoader(dataset, batch_size=arguments.batch_size, num_workers=0)
    paired_dataset = (
        TrajectoryShards(paired_directory, arguments.seed)
        if paired_directory is not None else None
    )
    paired_loader = (
        DataLoader(paired_dataset, batch_size=arguments.batch_size, num_workers=0)
        if paired_dataset is not None else None
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=arguments.batch_size, num_workers=0
    )

    started = time.perf_counter()
    model.train()
    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, arguments.epochs):
        dataset.set_epoch(epoch)
        if paired_dataset is not None:
            paired_dataset.set_epoch(epoch)
        paired_iterator = iter(paired_loader) if paired_loader is not None else None
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index <= resume_batch:
                continue
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["loss_mask"].to(device)
            output = model(input_ids, return_method_states=arguments.mechanism == "trajectory")
            logits = output["logits"]
            ce_loss = masked_mean(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels), mask)
            behavior_loss = torch.zeros((), device=device)
            state_loss = torch.zeros((), device=device)
            delta_loss = torch.zeros((), device=device)
            contrast_loss = torch.zeros((), device=device)
            contrast_metrics = {"pairs": 0, "student_js": 0.0, "teacher_js": 0.0}
            paired_loss = torch.zeros((), device=device)
            response_start_loss = torch.zeros((), device=device)

            if arguments.mechanism in ("distill", "trajectory"):
                behavior_loss = sparse_topk_kl(
                    logits,
                    batch["top_indices"].to(device),
                    batch["top_values"].to(device),
                    batch["logsumexp"].to(device),
                    mask=mask,
                )
            if arguments.mechanism == "trajectory":
                trajectory_losses = alignment(
                    output["method_states"], batch["trajectory"].to(device), mask=mask
                )
                state_loss = trajectory_losses["state"]
                delta_loss = trajectory_losses["delta"]
            if arguments.contrast_weight:
                contrast_function = (
                    conditional_assignment_loss
                    if arguments.contrast_objective == "assignment"
                    else conditional_contrast_loss
                )
                contrast_loss, contrast_metrics = contrast_function(
                    logits,
                    batch["top_indices"].to(device),
                    batch["top_values"].to(device),
                    batch["logsumexp"].to(device),
                    mask,
                )
            if paired_iterator is not None:
                paired_batch = next(paired_iterator)
                paired_mask = paired_batch["loss_mask"].to(device)
                paired_logits = model(paired_batch["input_ids"].to(device))["logits"]
                paired_ce = masked_mean(F.cross_entropy(
                    paired_logits.reshape(-1, paired_logits.shape[-1]),
                    paired_batch["labels"].to(device).reshape(-1),
                    reduction="none",
                ).reshape_as(paired_mask), paired_mask)
                paired_kl = sparse_topk_kl(
                    paired_logits,
                    paired_batch["top_indices"].to(device),
                    paired_batch["top_values"].to(device),
                    paired_batch["logsumexp"].to(device),
                    mask=paired_mask,
                )
                paired_loss = paired_ce + arguments.distillation_weight * paired_kl
            if arguments.response_start_weight:
                start_mask = torch.zeros_like(mask)
                for row in range(mask.shape[0]):
                    positions = torch.nonzero(mask[row] > 0, as_tuple=False).flatten()
                    start_mask[row, positions[: arguments.response_start_tokens]] = 1
                start_ce = masked_mean(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape_as(labels), start_mask)
                start_kl = sparse_topk_kl(
                    logits,
                    batch["top_indices"].to(device),
                    batch["top_values"].to(device),
                    batch["logsumexp"].to(device),
                    mask=start_mask,
                )
                response_start_loss = (
                    start_ce + arguments.distillation_weight * start_kl
                )

            loss = (
                ce_loss
                + arguments.distillation_weight * behavior_loss
                + arguments.state_weight * state_loss
                + arguments.delta_weight * delta_loss
                + arguments.contrast_weight * contrast_loss
                + arguments.paired_weight * paired_loss
                + arguments.response_start_weight * response_start_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, arguments.gradient_clip)
            optimizer.step()
            step += 1
            record = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.item()),
                "ce": float(ce_loss.item()),
                "distillation": float(behavior_loss.item()),
                "state": float(state_loss.item()),
                "delta": float(delta_loss.item()),
                "contrast": float(contrast_loss.item()),
                "contrast_pairs": contrast_metrics["pairs"],
                "contrast_student_js": contrast_metrics.get("student_js", 0.0),
                "contrast_teacher_js": contrast_metrics.get("teacher_js", 0.0),
                "contrast_assignment_accuracy": contrast_metrics.get(
                    "assignment_accuracy", 0.0
                ),
                "contrast_margin": contrast_metrics.get("margin", 0.0),
                "paired_correction": float(paired_loss.item()),
                "response_start": float(response_start_loss.item()),
            }
            history.append(record)
            if step == 1 or step % arguments.log_every == 0:
                print(json.dumps(record), flush=True)
            if arguments.checkpoint_every and step % arguments.checkpoint_every == 0:
                torch.save({
                    "model": model.state_dict(),
                    "alignment": alignment.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "step": step,
                    "history": history,
                }, output_directory / "observer-d-lm-resume.pt")
            if arguments.snapshot_every and step % arguments.snapshot_every == 0:
                torch.save({
                    "model": model.state_dict(),
                    "alignment": alignment.state_dict(),
                    "config": config.__dict__,
                    "step": step,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "source_manifests": [
                        str(training_directory / "manifest.json")
                        for training_directory in training_directories
                    ],
                }, output_directory / f"observer-d-lm-step-{step:06d}.pt")
            if arguments.max_steps and step >= arguments.max_steps:
                break
        if arguments.max_steps and step >= arguments.max_steps:
            break

    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    validation = evaluate(
        model, alignment, validation_loader, arguments.mechanism, device
    )
    if validation["teacher_ce"] is not None:
        uniform_ce = math.log(config.vocab_size)
        validation["normalized_ce_retention"] = (
            (uniform_ce - validation["ce"])
            / (uniform_ce - validation["teacher_ce"])
        )
        validation["perplexity_ratio_to_teacher"] = math.exp(
            validation["ce"] - validation["teacher_ce"]
        )
    result = {
        "mechanism": arguments.mechanism,
        "steps": step,
        "elapsed_seconds": elapsed,
        "parameter_count": model.parameter_count(),
        "teacher_parameter_count": manifest["model_parameter_count"],
        "parameter_compression": manifest["model_parameter_count"] / model.parameter_count(),
        "tokens": step * arguments.batch_size * manifest["config"]["sequence_length"],
        "final": history[-1] if history else None,
        "validation": validation,
        "training_shards": [path.name for path in train_shards],
        "validation_shards": [path.name for path in validation_shards],
        "history": history,
        "source_manifests": [
            str(training_directory / "manifest.json")
            for training_directory in training_directories
        ],
        "validation_manifest": str(validation_directory / "manifest.json"),
    }
    torch.save(
        {
            "model": model.state_dict(),
            "alignment": alignment.state_dict(),
            "config": config.__dict__,
            "result": result,
        },
        output_directory / f"observer-d-lm-{arguments.mechanism}.pt",
    )
    (output_directory / f"observer-d-lm-{arguments.mechanism}.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--additional-data", nargs="*", default=[])
    parser.add_argument("--paired-data")
    parser.add_argument("--paired-weight", type=float, default=0.0)
    parser.add_argument("--response-start-tokens", type=int, default=8)
    parser.add_argument("--response-start-weight", type=float, default=0.0)
    parser.add_argument("--validation-data")
    parser.add_argument("--mechanism", choices=("direct", "distill", "trajectory"), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--holdout-shards", type=int, default=1)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--attention-every", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=6)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--output-head-rank", type=int, default=0)
    parser.add_argument("--output-bias", action="store_true")
    parser.add_argument("--freeze-output-table", action="store_true")
    parser.add_argument("--freeze-output-projection", action="store_true")
    parser.add_argument("--attention-residual-scale", type=float, default=1.0)
    parser.add_argument("--ffn-residual-scale", type=float, default=1.0)
    parser.add_argument("--depth-adapter-rank", type=int, default=0)
    parser.add_argument("--input-embedding-rank", type=int, default=0)
    parser.add_argument("--method-cell-count", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--state-weight", type=float, default=0.25)
    parser.add_argument("--delta-weight", type=float, default=0.25)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    parser.add_argument(
        "--contrast-objective", choices=("js", "assignment"), default="js"
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--snapshot-every", type=int, default=0,
        help="Write immutable model-only measurement checkpoints at this step interval",
    )
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", default="research/results/lfm_training")
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    # The full per-step history is already durable in the result JSON. Avoid
    # duplicating thousands of records into interactive shell output.
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))
