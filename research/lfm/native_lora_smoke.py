#!/usr/bin/env python3
"""One native-model optimizer step with frozen-base and adapter reload audits."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.lfm.evaluate_pretrained_development import synchronize
from research.lfm.native_lora import adapter_state, install_lora, load_adapter, supervised_chat
from research.lfm.probe_lfm import choose_device


MODEL = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = choose_device(args.device)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    # FP32 smoke test avoids conflating mixed-precision failures with graph support.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32
    ).to(device).eval()
    row = supervised_chat(tokenizer,
        "Return only a JSON object mapping item to lamp and quantity to 3.",
        '{"item":"lamp","quantity":3}', 128)
    inputs = torch.tensor([row["input_ids"]], device=device)
    labels = torch.tensor([row["labels"]], device=device)
    with torch.no_grad():
        original = model(inputs, use_cache=False).logits.detach().clone()
    targets = install_lora(model)
    frozen_before = frozen_digest(model)
    initial_state = adapter_state(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
    with torch.no_grad():
        initialized = model(inputs, use_cache=False).logits
    initial_error = float((original - initialized).abs().max())
    del original, initialized
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = model(inputs, labels=labels, use_cache=False).loss
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite native loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    frozen_after = frozen_digest(model)
    final_state = adapter_state(model)
    update_norm = sum(float((final_state[n] - initial_state[n]).square().sum())
                      for n in final_state) ** 0.5
    with torch.no_grad():
        trained_logits = model(inputs, use_cache=False).logits.detach().clone()
    payload = {
        "model": MODEL, "revision": REVISION, "rank": 8, "alpha": 16,
        "targets": targets, "adapter": final_state, "optimizer": optimizer.state_dict(),
        "step": 1, "seed": 0, "cpu_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, output / "adapter-step-1.pt")
    load_adapter(model, initial_state)
    saved = torch.load(output / "adapter-step-1.pt", weights_only=True, map_location="cpu")
    load_adapter(model, saved["adapter"])
    with torch.no_grad():
        restored_logits = model(inputs, use_cache=False).logits
    reload_error = float((trained_logits - restored_logits).abs().max())
    resumed_optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    optimizer_steps = sorted({int(state["step"].item()) for state in resumed_optimizer.state.values()})
    report = {
        "scope": "one-example engineering smoke test, not capability training",
        "model": MODEL, "revision": REVISION, "device": device, "dtype": "float32",
        "targets": targets, "trainable_parameters": sum(p.numel() for p in parameters),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "input_ids": row["input_ids"], "labels": row["labels"],
        "supervised_tokens": row["supervised_tokens"],
        "loss_before_update": float(loss.detach()), "unclipped_gradient_norm": float(norm),
        "adapter_update_norm": update_norm, "initial_logit_max_error": initial_error,
        "reload_logit_max_error": reload_error,
        "frozen_before_sha256": frozen_before, "frozen_after_sha256": frozen_after,
        "frozen_parameters_have_no_grad": all(p.grad is None for p in model.parameters() if not p.requires_grad),
        "restored_optimizer_steps": optimizer_steps, "update_seconds": elapsed,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if device == "mps" else None,
        "memory_note": "MPS current allocation is not peak or total process memory",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__},
    }
    report["passed"] = (initial_error == 0 and reload_error == 0 and update_norm > 0
        and frozen_before == frozen_after and report["frozen_parameters_have_no_grad"]
        and optimizer_steps == [1])
    (output / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Native smoke-test gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    run(parser.parse_args())
