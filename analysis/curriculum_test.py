#!/usr/bin/env python3
"""
Phase 0: Curriculum Test
========================
Validates that temporal structure exists in training before building online observer.

Key question: Do both models learn in the same order (easy cases before hard cases)?

If learning curves correlate > 0.7, temporal structure exists and online observation
might capture meaningful information that post-hoc analysis cannot.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr, spearmanr
import json
import math

# Fine structure constant (matching original training)
ALPHA = 1.0 / 137.0


def compute_carry_count(a_bits, b_bits, carry_in):
    """Count carry operations during addition."""
    carries = 0
    carry = carry_in
    for i in range(4):
        bit_sum = a_bits[i] + b_bits[i] + carry
        if bit_sum >= 2:
            carries += 1
            carry = 1
        else:
            carry = 0
    return carries


def generate_dataset_with_carry():
    """Generate all 512 inputs with carry_count labels."""
    inputs = []
    outputs = []
    carry_counts = []

    for a in range(16):
        for b in range(16):
            for carry_in in [0, 1]:
                # Input bits
                a_bits = [(a >> i) & 1 for i in range(4)]
                b_bits = [(b >> i) & 1 for i in range(4)]

                # Compute true output
                total = a + b + carry_in
                s_bits = [(total >> i) & 1 for i in range(4)]
                carry_out = (total >> 4) & 1

                inputs.append(a_bits + b_bits + [carry_in])
                outputs.append(s_bits + [carry_out])
                carry_counts.append(compute_carry_count(a_bits, b_bits, carry_in))

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(outputs, dtype=torch.float32),
        np.array(carry_counts)
    )


class Monolithic4BitAdder(nn.Module):
    """Monolithic architecture (copy from original)."""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5),
        )

    def forward(self, x):
        return self.net(x)


class FullAdderBit(nn.Module):
    """Single bit full adder."""
    def __init__(self, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        return self.net(x)


class Compositional4BitAdder(nn.Module):
    """Compositional architecture (copy from original)."""
    def __init__(self, hidden_size=16):
        super().__init__()
        self.bit0_adder = FullAdderBit(hidden_size)
        self.bit1_adder = FullAdderBit(hidden_size)
        self.bit2_adder = FullAdderBit(hidden_size)
        self.bit3_adder = FullAdderBit(hidden_size)

    def forward(self, x):
        a = x[:, 0:4]
        b = x[:, 4:8]
        carry_in = x[:, 8:9]

        bit0_input = torch.cat([a[:, 0:1], b[:, 0:1], carry_in], dim=1)
        bit0_output = self.bit0_adder(bit0_input)
        s0, c0 = bit0_output[:, 0:1], bit0_output[:, 1:2]

        bit1_input = torch.cat([a[:, 1:2], b[:, 1:2], c0], dim=1)
        bit1_output = self.bit1_adder(bit1_input)
        s1, c1 = bit1_output[:, 0:1], bit1_output[:, 1:2]

        bit2_input = torch.cat([a[:, 2:3], b[:, 2:3], c1], dim=1)
        bit2_output = self.bit2_adder(bit2_input)
        s2, c2 = bit2_output[:, 0:1], bit2_output[:, 1:2]

        bit3_input = torch.cat([a[:, 3:4], b[:, 3:4], c2], dim=1)
        bit3_output = self.bit3_adder(bit3_input)
        s3, c3 = bit3_output[:, 0:1], bit3_output[:, 1:2]

        return torch.cat([s0, s1, s2, s3, c3], dim=1)


def get_scheduler(optimizer, num_warmup_steps, num_training_steps, peak_lr, min_lr):
    """Cosine schedule with warmup."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / peak_lr, cosine_decay)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_per_carry_accuracy(model, inputs, outputs, carry_counts):
    """Compute accuracy for each carry_count level."""
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
        pred_bits = torch.round(preds).long()
        true_bits = outputs.long()
        correct = (pred_bits == true_bits).all(dim=1).numpy()

    # Group by carry_count
    acc_by_carry = {}
    for c in range(5):  # 0-4 carries possible
        mask = carry_counts == c
        if mask.sum() > 0:
            acc_by_carry[c] = correct[mask].mean()
        else:
            acc_by_carry[c] = 0.0

    return acc_by_carry


def train_with_curriculum_tracking(model, inputs, outputs, carry_counts, epochs=200, name="model"):
    """Train model while tracking per-carry accuracy at each epoch."""

    optimizer = optim.Adam(model.parameters(), lr=ALPHA)
    criterion = nn.MSELoss()

    batch_size = 32
    n_batches = len(inputs) // batch_size
    num_training_steps = n_batches * epochs
    num_warmup_steps = n_batches * 10

    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps, ALPHA, ALPHA/1000)

    # Track per-carry accuracy at each epoch
    curriculum_log = []

    for epoch in range(epochs):
        # Train one epoch
        model.train()
        indices = torch.randperm(len(inputs))

        for i in range(0, len(inputs), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = inputs[batch_idx]
            batch_y = outputs[batch_idx]

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Evaluate per-carry accuracy
        acc_by_carry = compute_per_carry_accuracy(model, inputs, outputs, carry_counts)
        curriculum_log.append(acc_by_carry)

        if (epoch + 1) % 20 == 0:
            overall_acc = sum(acc_by_carry.values()) / 5
            print(f"  {name} Epoch {epoch+1:3d}: "
                  f"c0={acc_by_carry[0]:.2f} c1={acc_by_carry[1]:.2f} "
                  f"c2={acc_by_carry[2]:.2f} c3={acc_by_carry[3]:.2f} "
                  f"c4={acc_by_carry[4]:.2f} | avg={overall_acc:.2f}")

    return curriculum_log


def analyze_curriculum_correlation(mono_log, comp_log):
    """Analyze correlation between learning curves."""

    epochs = len(mono_log)

    print("\n" + "=" * 80)
    print("CURRICULUM CORRELATION ANALYSIS")
    print("=" * 80)

    # Convert to arrays: (epochs, 5) for each carry level
    mono_curves = np.array([[mono_log[e][c] for c in range(5)] for e in range(epochs)])
    comp_curves = np.array([[comp_log[e][c] for c in range(5)] for e in range(epochs)])

    # Per-carry-level correlation
    print("\nPer-Carry Learning Curve Correlation:")
    print("-" * 50)

    correlations = {}
    for c in range(5):
        mono_curve = mono_curves[:, c]
        comp_curve = comp_curves[:, c]

        # Skip if either curve is constant (no variance)
        if mono_curve.std() < 0.01 or comp_curve.std() < 0.01:
            print(f"  Carry {c}: SKIPPED (constant curve)")
            correlations[c] = np.nan
            continue

        r, p = pearsonr(mono_curve, comp_curve)
        correlations[c] = r
        print(f"  Carry {c}: r = {r:.4f} (p = {p:.4f})")

    # Overall correlation (flatten all curves)
    mono_flat = mono_curves.flatten()
    comp_flat = comp_curves.flatten()
    overall_r, overall_p = pearsonr(mono_flat, comp_flat)

    print(f"\nOverall correlation: r = {overall_r:.4f} (p = {overall_p:.6f})")

    # Learning order analysis
    print("\n" + "-" * 50)
    print("Learning Order Analysis:")
    print("-" * 50)

    # Find epoch where each carry level first exceeds 80% accuracy
    def first_above_threshold(curve, threshold=0.8):
        for i, acc in enumerate(curve):
            if acc >= threshold:
                return i
        return len(curve)  # Never reached

    print("\nEpoch to reach 80% accuracy:")
    print(f"  {'Carry':<10} {'Mono':>10} {'Comp':>10} {'Diff':>10}")
    print(f"  {'-'*40}")

    for c in range(5):
        mono_epoch = first_above_threshold(mono_curves[:, c])
        comp_epoch = first_above_threshold(comp_curves[:, c])
        diff = comp_epoch - mono_epoch

        mono_str = str(mono_epoch) if mono_epoch < epochs else ">200"
        comp_str = str(comp_epoch) if comp_epoch < epochs else ">200"

        print(f"  {c:<10} {mono_str:>10} {comp_str:>10} {diff:>+10}")

    # Check if both learn easy (low carry) before hard (high carry)
    mono_order = [first_above_threshold(mono_curves[:, c]) for c in range(5)]
    comp_order = [first_above_threshold(comp_curves[:, c]) for c in range(5)]

    mono_sorted = np.argsort(mono_order)
    comp_sorted = np.argsort(comp_order)

    order_corr, _ = spearmanr(mono_sorted, comp_sorted)
    print(f"\nLearning order correlation (Spearman): {order_corr:.4f}")

    return {
        'per_carry_correlations': correlations,
        'overall_correlation': overall_r,
        'overall_p_value': overall_p,
        'learning_order_correlation': order_corr,
        'mono_curves': mono_curves.tolist(),
        'comp_curves': comp_curves.tolist(),
    }


def main():
    print("=" * 80)
    print("PHASE 0: CURRICULUM TEST")
    print("Validating temporal structure before building online observer")
    print("=" * 80)

    # Generate data
    print("\nGenerating dataset...")
    inputs, outputs, carry_counts = generate_dataset_with_carry()
    print(f"  Total samples: {len(inputs)}")
    print(f"  Carry count distribution: {np.bincount(carry_counts)}")

    # Train monolithic
    print("\n" + "-" * 80)
    print("Training MONOLITHIC model with curriculum tracking...")
    print("-" * 80)
    mono_model = Monolithic4BitAdder(hidden_size=64)
    mono_log = train_with_curriculum_tracking(
        mono_model, inputs, outputs, carry_counts,
        epochs=200, name="Mono"
    )

    # Train compositional
    print("\n" + "-" * 80)
    print("Training COMPOSITIONAL model with curriculum tracking...")
    print("-" * 80)
    comp_model = Compositional4BitAdder(hidden_size=16)
    comp_log = train_with_curriculum_tracking(
        comp_model, inputs, outputs, carry_counts,
        epochs=200, name="Comp"
    )

    # Analyze correlation
    results = analyze_curriculum_correlation(mono_log, comp_log)

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    threshold = 0.7
    overall_r = results['overall_correlation']

    if overall_r > threshold:
        print(f"\n✓ TEMPORAL STRUCTURE EXISTS")
        print(f"  Overall correlation: {overall_r:.4f} > {threshold}")
        print(f"  Both models learn in correlated patterns.")
        print(f"  → Proceed with online observer implementation.")
    elif overall_r > 0.5:
        print(f"\n⚠ MODERATE TEMPORAL STRUCTURE")
        print(f"  Overall correlation: {overall_r:.4f}")
        print(f"  Some shared learning patterns exist.")
        print(f"  → Proceed with caution.")
    else:
        print(f"\n✗ WEAK TEMPORAL STRUCTURE")
        print(f"  Overall correlation: {overall_r:.4f} < 0.5")
        print(f"  Models learn independently.")
        print(f"  → Online observer may not help.")

    # Save results
    results_path = 'journal/curriculum_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
