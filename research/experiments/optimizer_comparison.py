#!/usr/bin/env python3
"""AdamW/Muon comparison for detached Delta Observer students.

This is an initial mechanism test, not a claim of optimizer superiority.  It
trains independent source models, captures their trajectories, learns a
label-free trajectory autoencoder, and then compares detached students under
matched update and example budgets.

Run from the repository root:

    python -m research.experiments.optimizer_comparison --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.delta_observer import Compositional4BitAdder, Monolithic4BitAdder, generate_dataset
from research.observers import ObserverDNeon
from research.optimizers import HybridMuon


@dataclass
class Config:
    seeds: list[int]
    source_epochs: int = 150
    observer_epochs: int = 100
    student_epochs: int = 150
    batch_size: int = 64
    checkpoints: int = 10
    latent_dim: int = 32
    representation_weight: float = 0.25
    target_accuracy: float = 0.95
    adamw_lr: float = 3e-3
    muon_lr: float = 2e-2
    weight_decay: float = 1e-2
    observer_backend: str = "neon"
    output: str = "research/results/optimizer_comparison.json"


class ObserverD(nn.Module):
    """Observer.D compresses paired source states and changes without labels."""

    def __init__(self, input_dim: int = 257, state_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(approximate="tanh"), nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(approximate="tanh"), nn.Linear(128, state_dim)
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(features)
        return latent, self.decoder(latent)


class Student(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.input = nn.Linear(9, 64)
        self.representation = nn.Linear(64, latent_dim)
        self.output = nn.Linear(latent_dim, 5)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.input(inputs))
        representation = self.representation(hidden)
        return self.output(F.gelu(representation)), representation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = (logits.sigmoid() >= 0.5).to(targets.dtype)
    return float((predictions == targets).all(dim=1).float().mean().item())


def make_split(seed: int, n: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    return {"train": order[:384], "validation": order[384:448], "heldout": order[448:]}


def split_hash(split: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in ("train", "validation", "heldout"):
        digest.update(name.encode())
        digest.update(split[name].numpy().tobytes())
    return digest.hexdigest()


def batches(indices: torch.Tensor, batch_size: int, generator: torch.Generator):
    shuffled = indices[torch.randperm(len(indices), generator=generator)]
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def train_source(
    model: nn.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    train_indices: torch.Tensor,
    config: Config,
    seed: int,
) -> tuple[nn.Module, list[torch.Tensor], float]:
    """Train one source and return train-set activation checkpoints."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=config.weight_decay)
    checkpoint_epochs = set(np.linspace(0, config.source_epochs - 1, config.checkpoints, dtype=int).tolist())
    snapshots = []
    start = time.perf_counter()

    model.eval()
    with torch.no_grad():
        snapshots.append(model.get_activations(inputs[train_indices]).detach().cpu())

    for epoch in range(config.source_epochs):
        model.train()
        for index in batches(train_indices, config.batch_size, generator):
            logits = model(inputs[index])
            loss = F.binary_cross_entropy_with_logits(logits, outputs[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in checkpoint_epochs:
            model.eval()
            with torch.no_grad():
                snapshots.append(model.get_activations(inputs[train_indices]).detach().cpu())
    return model.eval(), snapshots, time.perf_counter() - start


def build_trajectory_features(
    mono_snapshots: list[torch.Tensor], comp_snapshots: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    count = min(len(mono_snapshots), len(comp_snapshots))
    mono = torch.stack(mono_snapshots[:count])
    comp = torch.stack(comp_snapshots[:count])
    state = torch.cat((mono, comp), dim=-1)
    delta = torch.zeros_like(state)
    delta[1:] = state[1:] - state[:-1]
    step = torch.linspace(0.0, 1.0, count).view(count, 1, 1).expand(-1, state.shape[1], -1)
    features = torch.cat((state, delta, step), dim=-1)
    targets = torch.cat((state, delta), dim=-1)
    return features, targets, count, state.shape[1]


def fit_observer(
    mono_snapshots: list[torch.Tensor],
    comp_snapshots: list[torch.Tensor],
    config: Config,
    seed: int,
) -> tuple[torch.Tensor, dict, float]:
    raw_features, raw_targets, steps, samples = build_trajectory_features(mono_snapshots, comp_snapshots)
    features = raw_features.reshape(-1, raw_features.shape[-1])
    targets = raw_targets.reshape(-1, raw_targets.shape[-1])
    feature_mean, feature_std = features.mean(0), features.std(0).clamp_min(1e-5)
    target_mean, target_std = targets.mean(0), targets.std(0).clamp_min(1e-5)
    features = (features - feature_mean) / feature_std
    targets = (targets - target_mean) / target_std

    observer = ObserverD(features.shape[1], targets.shape[1], config.latent_dim)
    optimizer = torch.optim.AdamW(observer.parameters(), lr=3e-3, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    all_indices = torch.arange(len(features))
    start = time.perf_counter()
    final_loss = float("nan")
    for _ in range(config.observer_epochs):
        observer.train()
        for index in batches(all_indices, config.batch_size, generator):
            _, reconstruction = observer(features[index])
            loss = F.mse_loss(reconstruction, targets[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

    observer.eval()
    with torch.no_grad():
        reference_start = time.perf_counter()
        reference_latent = observer.encoder(features)
        reference_seconds = time.perf_counter() - reference_start
        if config.observer_backend == "neon":
            native = ObserverDNeon()
            native_start = time.perf_counter()
            latent = native.encode(features, observer.encoder)
            native_seconds = time.perf_counter() - native_start
            max_error = float((latent - reference_latent).abs().max().item())
            if max_error > 1e-3:
                raise RuntimeError(f"Observer.D C/NEON parity failure: max error {max_error:.6g}")
            backend = native.backend
        else:
            latent = reference_latent
            native_seconds = None
            max_error = 0.0
            backend = "pytorch"
        latent = latent.reshape(steps, samples, -1)
        # A stable per-example learning signature using every chronological state.
        signature = latent.mean(dim=0)
        signature = (signature - signature.mean(0)) / signature.std(0).clamp_min(1e-5)
    metadata = {
        "name": "Observer.D",
        "backend": backend,
        "final_reconstruction_loss": final_loss,
        "trajectory_steps": steps,
        "signature_rows": len(features),
        "pytorch_encode_seconds": reference_seconds,
        "native_encode_seconds": native_seconds,
        "native_max_absolute_error": max_error,
    }
    return signature.detach(), metadata, time.perf_counter() - start


def optimizer_state_bytes(optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def evaluate(model: Student, inputs: torch.Tensor, outputs: torch.Tensor, indices: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits, _ = model(inputs[indices])
        return {
            "loss": float(F.binary_cross_entropy_with_logits(logits, outputs[indices]).item()),
            "exact_accuracy": exact_accuracy(logits, outputs[indices]),
            "bit_accuracy": float(((logits.sigmoid() >= 0.5) == outputs[indices].bool()).float().mean().item()),
        }


def train_student(
    mechanism: str,
    optimizer_name: str,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    split: dict[str, torch.Tensor],
    teacher_probabilities: torch.Tensor,
    observer_signature: torch.Tensor,
    config: Config,
    seed: int,
) -> tuple[dict, list[dict]]:
    set_seed(seed)
    model = Student(config.latent_dim)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.adamw_lr, weight_decay=config.weight_decay
        )
        grouping = {"adamw": [name for name, _ in model.named_parameters()]}
    else:
        optimizer = HybridMuon(
            model,
            muon_lr=config.muon_lr,
            adamw_lr=config.adamw_lr,
            weight_decay=config.weight_decay,
        )
        grouping = {"muon": optimizer.muon_names, "adamw": optimizer.adamw_names}

    generator = torch.Generator().manual_seed(seed + 10_000)
    train_indices = split["train"]
    curve = []
    examples_seen = 0
    target_epoch = None
    start = time.perf_counter()
    for epoch in range(config.student_epochs):
        model.train()
        for local_index in batches(torch.arange(len(train_indices)), config.batch_size, generator):
            global_index = train_indices[local_index]
            logits, representation = model(inputs[global_index])
            if mechanism in ("task", "task_observer"):
                output_loss = F.binary_cross_entropy_with_logits(logits, outputs[global_index])
            else:
                output_loss = F.binary_cross_entropy_with_logits(logits, teacher_probabilities[local_index])
            loss = output_loss
            if mechanism in ("task_observer", "observer_distill"):
                representation_loss = F.mse_loss(representation, observer_signature[local_index])
                loss = loss + config.representation_weight * representation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            examples_seen += len(local_index)

        # Validation controls time-to-target. Held-out data is touched only
        # once, after training, and cannot influence updates or model choice.
        validation = evaluate(model, inputs, outputs, split["validation"])
        curve.append({"epoch": epoch + 1, "examples_seen": examples_seen, **validation})
        if target_epoch is None and validation["exact_accuracy"] >= config.target_accuracy:
            target_epoch = epoch + 1

    elapsed = time.perf_counter() - start
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "mechanism": mechanism,
        "optimizer": optimizer_name,
        "validation": evaluate(model, inputs, outputs, split["validation"]),
        "heldout": evaluate(model, inputs, outputs, split["heldout"]),
        "validation_time_to_target_epoch": target_epoch,
        "student_seconds": elapsed,
        "training_examples": examples_seen,
        "update_steps": config.student_epochs * ((len(train_indices) + config.batch_size - 1) // config.batch_size),
        "parameter_count": parameters,
        "estimated_training_flops": 6 * parameters * examples_seen,
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "parameter_groups": grouping,
    }
    return result, curve


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_seed(seed: int, config: Config) -> dict:
    set_seed(seed)
    inputs, outputs, _, _ = generate_dataset()
    split = make_split(seed, len(inputs))

    mono, mono_states, mono_seconds = train_source(
        Monolithic4BitAdder(), inputs, outputs, split["train"], config, seed + 1
    )
    comp, comp_states, comp_seconds = train_source(
        Compositional4BitAdder(), inputs, outputs, split["train"], config, seed + 2
    )
    with torch.no_grad():
        teacher_probabilities = (
            mono(inputs[split["train"]]).sigmoid() + comp(inputs[split["train"]]).sigmoid()
        ) / 2.0
        source_metrics = {
            "monolithic": evaluate_source(mono, inputs, outputs, split),
            "compositional": evaluate_source(comp, inputs, outputs, split),
        }

    signature, observer_metadata, observer_seconds = fit_observer(
        mono_states, comp_states, config, seed + 3
    )

    runs = []
    curves = {}
    for mechanism in ("task", "task_observer", "distill", "observer_distill"):
        for optimizer_name in ("adamw", "muon"):
            run_id = f"seed-{seed}_{mechanism}_{optimizer_name}"
            result, curve = train_student(
                mechanism,
                optimizer_name,
                inputs,
                outputs,
                split,
                teacher_probabilities,
                signature,
                config,
                seed + 100,
            )
            result["run_id"] = run_id
            result["seed"] = seed
            result["split_hash"] = split_hash(split)
            runs.append(result)
            curves[run_id] = curve

    return {
        "seed": seed,
        "split_hash": split_hash(split),
        "source_metrics": source_metrics,
        "source_training_seconds": mono_seconds + comp_seconds,
        "observer_training_seconds": observer_seconds,
        "observer": observer_metadata,
        "student_runs": runs,
        "learning_curves": curves,
    }


def evaluate_source(model, inputs, outputs, split):
    result = {}
    model.eval()
    with torch.no_grad():
        for name, indices in split.items():
            logits = model(inputs[indices])
            result[name] = {"exact_accuracy": exact_accuracy(logits, outputs[indices])}
    return result


def summarize(seed_results: list[dict]) -> dict:
    grouped = {}
    indexed = {}
    for seed_result in seed_results:
        for run in seed_result["student_runs"]:
            key = f"{run['mechanism']}+{run['optimizer']}"
            grouped.setdefault(key, {"exact": [], "bit": [], "seconds": []})
            grouped[key]["exact"].append(run["heldout"]["exact_accuracy"])
            grouped[key]["bit"].append(run["heldout"]["bit_accuracy"])
            grouped[key]["seconds"].append(run["student_seconds"])
            indexed[(seed_result["seed"], run["mechanism"], run["optimizer"])] = run

    def statistics(values):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        # Normal interval is descriptive here; retain per-seed values for any
        # preferred bootstrap or paired analysis.
        margin = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return {
            "mean": mean,
            "std": std,
            "ci95": [mean - margin, mean + margin],
            "per_seed": values,
        }

    arms = {
        key: {
            "heldout_exact": statistics(values["exact"]),
            "heldout_bit": statistics(values["bit"]),
            "student_seconds": statistics(values["seconds"]),
        }
        for key, values in grouped.items()
    }

    contrasts = {}
    for optimizer_name in ("adamw", "muon"):
        for guided, baseline in (
            ("task_observer", "task"),
            ("observer_distill", "distill"),
        ):
            deltas = []
            for seed_result in seed_results:
                seed = seed_result["seed"]
                guided_run = indexed[(seed, guided, optimizer_name)]
                baseline_run = indexed[(seed, baseline, optimizer_name)]
                deltas.append(
                    guided_run["heldout"]["exact_accuracy"]
                    - baseline_run["heldout"]["exact_accuracy"]
                )
            contrasts[f"{guided}-{baseline}+{optimizer_name}"] = statistics(deltas)
    return {"arms": arms, "paired_observer_effects": contrasts}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--source-epochs", type=int, default=150)
    parser.add_argument("--observer-epochs", type=int, default=100)
    parser.add_argument("--student-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoints", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--representation-weight", type=float, default=0.25)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--adamw-lr", type=float, default=3e-3)
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--observer-backend", choices=("neon", "pytorch"), default="neon")
    parser.add_argument("--output", default="research/results/optimizer_comparison.json")
    arguments = parser.parse_args()
    return Config(**vars(arguments))


def main() -> None:
    config = parse_args()
    started = time.time()
    seed_results = []
    for seed in config.seeds:
        print(f"Running seed {seed} ...", flush=True)
        seed_results.append(run_seed(seed, config))

    report = {
        "experiment_id": "Observer.D-DOR-001-phase4-initial",
        "method": "Observer.D",
        "git_commit": git_commit(),
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "supervision": {
            "task": "student receives ground-truth outputs",
            "task_observer": "student receives ground truth and label-free trajectory signature",
            "distill": "student receives final A/B ensemble outputs",
            "observer_distill": "student receives ensemble outputs and label-free trajectory signature",
            "observer_receives_ground_truth": False,
        },
        "elapsed_seconds": time.time() - started,
        "seeds": seed_results,
        "summary": summarize(seed_results),
    }
    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
