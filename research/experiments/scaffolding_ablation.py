#!/usr/bin/env python3
"""One-factor-at-a-time ablation of transient clustering.

Compares the paper's online Delta Observer pipeline (models/delta_observer.py)
with the simplified notebook pipeline (notebooks/99_full_reproduction.ipynb),
flipping one difference at a time. Protocol:
research/reports/scaffolding_ablation_protocol.md

Run from the repository root, in the paper environment (root requirements.txt):

    python -m research.experiments.scaffolding_ablation --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from models.delta_observer import ALPHA, generate_dataset, get_scheduler

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research/reports/scaffolding_ablation_protocol.md"

FACTORS = ("data", "loss", "schedule", "mono", "comp", "observer", "batch")
PAPER = {"data": "512", "loss": "mse", "schedule": "alpha", "mono": "paper",
         "comp": "paper", "observer": "paper", "batch": 32}
NOTEBOOK = {"data": "256", "loss": "bce", "schedule": "const", "mono": "notebook",
            "comp": "notebook", "observer": "notebook", "batch": 64}

PRESENCE_PEAK = 0.10
PRESENCE_RATIO = 0.5
FINAL_WINDOW = 10


def configurations():
    """Both baselines plus every single-factor flip in both directions."""
    configs = {"P": dict(PAPER), "N": dict(NOTEBOOK)}
    for factor in FACTORS:
        configs[f"P/{factor}"] = {**PAPER, factor: NOTEBOOK[factor]}
        configs[f"N/{factor}"] = {**NOTEBOOK, factor: PAPER[factor]}
    return configs


def addendum_a_configurations():
    """Protocol addendum A: split the observer factor on the paper baseline."""
    v1 = configurations()
    return {"P": v1["P"], "P/observer": v1["P/observer"], "N": v1["N"],
            "P/obs-nobit": {**PAPER, "observer": "paper_nobit"},
            "P/obs-constlr": {**PAPER, "observer": "paper_constlr"}}


def addendum_b_configurations():
    """Protocol addendum B: an observer with no supervised heads."""
    a = addendum_a_configurations()
    return {"P": a["P"], "P/obs-nobit": a["P/obs-nobit"],
            "P/obs-unsup": {**PAPER, "observer": "paper_unsup"}, "N": a["N"]}


SUITES = {"v1": configurations, "addendum-a": addendum_a_configurations,
          "addendum-b": addendum_b_configurations}
SOURCE_MEASURES = ("src_concat", "src_mono", "src_comp", "src_pca16")


# =============================================================================
# Data
# =============================================================================

def make_data(variant):
    """512-case paper data, or its 256-case no-carry-in subset (8 inputs)."""
    inputs, outputs, carry_counts, bit_positions = generate_dataset()
    if variant == "512":
        return inputs, outputs, carry_counts, bit_positions
    keep = inputs[:, 8] == 0
    return inputs[keep][:, :8], outputs[keep], carry_counts[keep], bit_positions[keep]


# =============================================================================
# Models. Layer creation order mirrors the source pipelines so that seeded
# initialization matches them exactly.
# =============================================================================

class PaperMono(nn.Module):
    """models/delta_observer.py Monolithic4BitAdder; observer sees layer 1."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def activations(self, x):
        return F.relu(self.fc1(x))


class NotebookMono(nn.Module):
    """Notebook MonolithicMLP; observer sees layer 2. Returns logits."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 5)

    def forward(self, x):
        return self.fc3(self.activations(x))

    def activations(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))


def _carry_in(x):
    return x[:, 8:9] if x.shape[1] == 9 else x.new_zeros(x.shape[0], 1)


class PaperFullAdder(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class PaperComp(nn.Module):
    """models/delta_observer.py Compositional4BitAdder; observer sees each module's layer 1."""

    def __init__(self, hidden=16):
        super().__init__()
        self.bit0 = PaperFullAdder(hidden)
        self.bit1 = PaperFullAdder(hidden)
        self.bit2 = PaperFullAdder(hidden)
        self.bit3 = PaperFullAdder(hidden)

    def _run(self, x):
        carry, sums, acts = _carry_in(x), [], []
        for i, adder in enumerate((self.bit0, self.bit1, self.bit2, self.bit3)):
            bit_input = torch.cat([x[:, i:i + 1], x[:, i + 4:i + 5], carry], dim=1)
            acts.append(F.relu(adder.fc1(bit_input)))
            out = adder(bit_input)
            sums.append(out[:, 0:1])
            carry = out[:, 1:2]
        return torch.cat(sums + [carry], dim=1), torch.cat(acts, dim=1)

    def forward(self, x):
        return self._run(x)[0]

    def activations(self, x):
        return self._run(x)[1]


class NotebookComp(nn.Module):
    """Notebook CompositionalNetwork with learned carry heads. Returns logits."""

    def __init__(self, hidden=16):
        super().__init__()
        self.bit_modules = nn.ModuleList([
            nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            for _ in range(4)
        ])
        self.carry_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(4)])
        self.output = nn.Linear(4 * hidden, 5)

    def activations(self, x):
        carry, hidden = _carry_in(x), []
        for i in range(4):
            h = self.bit_modules[i](torch.cat([x[:, i:i + 1], x[:, i + 4:i + 5], carry], dim=1))
            hidden.append(h)
            carry = torch.sigmoid(self.carry_heads[i](h))
        return torch.cat(hidden, dim=1)

    def forward(self, x):
        return self.output(self.activations(x))


class Observer(nn.Module):
    """OnlineDeltaObserver (bit_head=True) or the notebook DeltaObserver."""

    def __init__(self, mono_dim=64, comp_dim=64, latent_dim=16, bit_head=True, carry_head=True):
        super().__init__()
        self.mono_encoder = nn.Sequential(nn.Linear(mono_dim, 32), nn.ReLU(), nn.Dropout(0.1))
        self.comp_encoder = nn.Sequential(nn.Linear(comp_dim, 32), nn.ReLU(), nn.Dropout(0.1))
        self.shared_encoder = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, latent_dim))
        self.mono_decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, mono_dim))
        self.comp_decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, comp_dim))
        self.bit_classifier = (nn.Sequential(nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, 4))
                               if bit_head else None)
        self.carry_regressor = (nn.Sequential(nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, 1))
                                if carry_head else None)

    def encode(self, mono_act, comp_act):
        joint = torch.cat([self.mono_encoder(mono_act), self.comp_encoder(comp_act)], dim=-1)
        return self.shared_encoder(joint)

    def loss(self, mono_act, comp_act, carry, bit):
        latent = self.encode(mono_act, comp_act)
        total = (F.mse_loss(self.mono_decoder(latent), mono_act) +
                 F.mse_loss(self.comp_decoder(latent), comp_act))
        if self.bit_classifier is not None:
            total = total + F.cross_entropy(self.bit_classifier(latent), bit)
        if self.carry_regressor is not None:
            total = total + 0.1 * F.mse_loss(self.carry_regressor(latent).squeeze(), carry.float())
        return total


# =============================================================================
# Training
# =============================================================================

def task_loss(kind, out, target):
    return F.mse_loss(out, target) if kind == "mse" else F.binary_cross_entropy_with_logits(out, target)


def accuracy(kind, out, target):
    pred = torch.round(out) if kind == "mse" else (out > 0).float()
    return float((pred == target).all(dim=1).float().mean())


def latent_metrics(latent, carry):
    r2 = LinearRegression().fit(latent, carry).score(latent, carry)
    return float(silhouette_score(latent, carry)), float(r2)


def train(config, seed, epochs=200, snapshots="sweep"):
    """Train all three models concurrently under config.

    snapshots="sweep" measures every epoch, in both observer modes, under an
    isolated RNG so measurement never perturbs training. "replicate"
    reproduces models/delta_observer.py exactly: its snapshot schedule, train
    mode only, sharing the training RNG. "none" takes no snapshots.
    """
    replicate = snapshots == "replicate"
    torch.manual_seed(seed)
    inputs, outputs, carry_counts, bit_positions = make_data(config["data"])
    n, in_dim, batch = len(inputs), inputs.shape[1], config["batch"]
    n_batches = n // batch

    mono = (PaperMono if config["mono"] == "paper" else NotebookMono)(in_dim)
    comp = (PaperComp if config["comp"] == "paper" else NotebookComp)()
    observer = Observer(bit_head=config["observer"] in ("paper", "paper_constlr"),
                        carry_head=config["observer"] != "paper_unsup")

    if config["schedule"] == "alpha":
        mono_opt = optim.Adam(mono.parameters(), lr=ALPHA)
        comp_opt = optim.Adam(comp.parameters(), lr=ALPHA)
        source_scheds = [get_scheduler(o, n_batches * 10, n_batches * epochs, ALPHA, ALPHA / 1000)
                         for o in (mono_opt, comp_opt)]
    else:
        mono_opt = optim.Adam(mono.parameters(), lr=0.01)
        comp_opt = optim.Adam(comp.parameters(), lr=0.01)
        source_scheds = []
    obs_opt = optim.Adam(observer.parameters(), lr=0.001)
    obs_sched = (optim.lr_scheduler.CosineAnnealingLR(obs_opt, epochs)
                 if config["observer"] in ("paper", "paper_nobit", "paper_unsup") else None)

    carry_np = carry_counts.numpy()
    history = {k: [] for k in ("sil_train", "sil_eval", "r2_train", "r2_eval", "mono_acc", "comp_acc",
                               "r2_src_pca16", *(f"sil_{m}" for m in SOURCE_MEASURES))}
    replica_snapshots = []

    for epoch in range(epochs):
        mono.train(); comp.train(); observer.train()
        indices = torch.randperm(n)
        for i in range(0, n, batch):
            idx = indices[i:i + batch]
            x, y = inputs[idx], outputs[idx]
            mono_loss = task_loss(config["loss"], mono(x), y)
            comp_loss = task_loss(config["loss"], comp(x), y)
            mono_opt.zero_grad(); mono_loss.backward(); mono_opt.step()
            if source_scheds: source_scheds[0].step()
            comp_opt.zero_grad(); comp_loss.backward(); comp_opt.step()
            if source_scheds: source_scheds[1].step()
            with torch.no_grad():
                mono_act, comp_act = mono.activations(x), comp.activations(x)
            obs_loss = observer.loss(mono_act, comp_act, carry_counts[idx], bit_positions[idx])
            obs_opt.zero_grad(); obs_loss.backward(); obs_opt.step()
        if obs_sched is not None:
            obs_sched.step()

        if replicate:
            if epoch % 10 == 0 or epoch < 20:
                with torch.no_grad():
                    replica_snapshots.append(
                        observer.encode(mono.activations(inputs), comp.activations(inputs)).numpy().copy())
            continue
        if snapshots == "none":
            continue

        with torch.random.fork_rng(devices=[]), torch.no_grad():
            mono_act, comp_act = mono.activations(inputs), comp.activations(inputs)
            latent_train = observer.encode(mono_act, comp_act).numpy()
            observer.eval()
            latent_eval = observer.encode(mono_act, comp_act).numpy()
            observer.train()
            history["mono_acc"].append(accuracy(config["loss"], mono(inputs), outputs))
            history["comp_acc"].append(accuracy(config["loss"], comp(inputs), outputs))
        for mode, latent in (("train", latent_train), ("eval", latent_eval)):
            sil, r2 = latent_metrics(latent, carry_np)
            history[f"sil_{mode}"].append(sil)
            history[f"r2_{mode}"].append(r2)
        # Observer-free: clustering in the sources' own (standardized) activations
        mono_np, comp_np = mono_act.numpy(), comp_act.numpy()
        concat = StandardScaler().fit_transform(np.concatenate([mono_np, comp_np], axis=1))
        for measure, acts in (("src_concat", concat), ("src_mono", StandardScaler().fit_transform(mono_np)),
                              ("src_comp", StandardScaler().fit_transform(comp_np))):
            history[f"sil_{measure}"].append(float(silhouette_score(acts, carry_np)))
        sil, r2 = latent_metrics(PCA(n_components=16).fit_transform(concat), carry_np)
        history["sil_src_pca16"].append(sil)
        history["r2_src_pca16"].append(r2)

    observer.eval(); mono.eval(); comp.eval()
    with torch.no_grad():
        mono_act, comp_act = mono.activations(inputs), comp.activations(inputs)
        final_latent = observer.encode(mono_act, comp_act).numpy()
    if snapshots != "sweep":
        return {"snapshots": replica_snapshots, "final_latent": final_latent}

    pca_latent = PCA(n_components=16).fit_transform(torch.cat([mono_act, comp_act], dim=1).numpy())
    pca_sil, pca_r2 = latent_metrics(pca_latent, carry_np)
    online_sil, online_r2 = latent_metrics(final_latent, carry_np)
    return {"history": history, "summary": summarize(history, online_r2, online_sil, pca_r2, pca_sil)}


def summarize(history, online_r2, online_sil, pca_r2, pca_sil):
    summary = {}
    for mode in ("train", "eval", *SOURCE_MEASURES):
        sil = np.array(history[f"sil_{mode}"])
        peak, final = float(sil.max()), float(sil[-FINAL_WINDOW:].mean())
        summary[mode] = {
            "peak_sil": peak, "peak_epoch": int(sil.argmax()), "final_sil": final,
            "present": bool(peak >= PRESENCE_PEAK and final < PRESENCE_RATIO * peak),
        }
    summary.update({
        "online_r2": online_r2, "online_sil": online_sil, "pca_r2": pca_r2, "pca_sil": pca_sil,
        "online_minus_pca": online_r2 - pca_r2,
        "mono_acc": history["mono_acc"][-1], "comp_acc": history["comp_acc"][-1],
    })
    return summary


# =============================================================================
# Sweep
# =============================================================================

def _run(job):
    name, config, seed, epochs = job
    torch.set_num_threads(1)
    start = time.perf_counter()
    result = train(config, seed, epochs)
    result.update({"config_name": name, "seed": seed, "seconds": time.perf_counter() - start})
    return result


def aggregate(runs, configs):
    table = {}
    for name in configs:
        rows = [r["summary"] for r in runs if r["config_name"] == name]
        entry = {"n": len(rows)}
        for mode in ("train", "eval", *SOURCE_MEASURES):
            entry[mode] = {
                "present": sum(r[mode]["present"] for r in rows),
                **{f"{k}_{stat}": float(fn([r[mode][k] for r in rows]))
                   for k in ("peak_sil", "final_sil") for stat, fn in (("mean", np.mean), ("sd", np.std))},
            }
        for k in ("online_r2", "pca_r2", "online_minus_pca"):
            entry[f"{k}_mean"] = float(np.mean([r[k] for r in rows]))
            entry[f"{k}_sd"] = float(np.std([r[k] for r in rows]))
        entry["min_source_acc"] = float(min(min(r["mono_acc"], r["comp_acc"]) for r in rows))
        table[name] = entry
    return table


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--suite", choices=sorted(SUITES), default="v1")
    parser.add_argument("--configs", nargs="+", default=None, help="subset of configuration names")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default=None,
                        help="default: research/results/scaffolding_ablation_<suite>.json")
    args = parser.parse_args()

    configs = SUITES[args.suite]()
    if args.configs:
        configs = {k: configs[k] for k in args.configs}
    jobs = [(name, cfg, seed, args.epochs) for name, cfg in configs.items() for seed in args.seeds]

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    with get_context("spawn").Pool(args.workers) as pool:
        runs = []
        for result in pool.imap_unordered(_run, jobs):
            runs.append(result)
            s = result["summary"]
            print(f"[{len(runs):3d}/{len(jobs)}] {result['config_name']:<12} seed {result['seed']} | "
                  f"peak {s['train']['peak_sil']:.3f} final {s['train']['final_sil']:.3f} "
                  f"(eval {s['eval']['final_sil']:.3f}) | online-PCA {s['online_minus_pca']:+.4f} | "
                  f"acc {s['mono_acc']:.2f}/{s['comp_acc']:.2f}", flush=True)
    runs.sort(key=lambda r: (list(configs).index(r["config_name"]), r["seed"]))

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    report = {
        "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(__file__), "git_commit": commit,
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "platform": platform.platform(), "threads_per_run": 1},
        "settings": {"suite": args.suite, "seeds": args.seeds, "epochs": args.epochs, "presence_peak": PRESENCE_PEAK,
                     "presence_ratio": PRESENCE_RATIO, "final_window": FINAL_WINDOW},
        "configurations": configs,
        "summary": aggregate(runs, configs),
        "runs": runs,
    }
    destination = ROOT / (args.output or f"research/results/scaffolding_ablation_{args.suite.replace('-', '_')}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=1))
    print(f"\nWrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
