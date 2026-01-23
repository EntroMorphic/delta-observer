#!/usr/bin/env python3
"""
Phase 1: Minimal Online Delta Observer
======================================
Trains the Delta Observer concurrently with both task models.

The observer sees activations at each batch, learning from the full
training trajectory rather than just the final state.

Key insight from curriculum test: Learning happens in epochs 1-20.
The observer must capture these early dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import math
from datetime import datetime


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


def generate_dataset():
    """Generate all 512 4-bit addition cases."""
    inputs = []
    outputs = []
    carry_counts = []
    bit_positions = []

    for a in range(16):
        for b in range(16):
            for carry_in in [0, 1]:
                a_bits = [(a >> i) & 1 for i in range(4)]
                b_bits = [(b >> i) & 1 for i in range(4)]

                total = a + b + carry_in
                s_bits = [(total >> i) & 1 for i in range(4)]
                carry_out = (total >> 4) & 1

                inputs.append(a_bits + b_bits + [carry_in])
                outputs.append(s_bits + [carry_out])
                carry_counts.append(compute_carry_count(a_bits, b_bits, carry_in))

                # Bit position: which bit has most "activity" (crude approximation)
                bit_positions.append(sum(s_bits[:4]) % 4)

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(outputs, dtype=torch.float32),
        torch.tensor(carry_counts, dtype=torch.long),
        torch.tensor(bit_positions, dtype=torch.long),
    )


# =============================================================================
# Models
# =============================================================================

class Monolithic4BitAdder(nn.Module):
    """Monolithic architecture with activation extraction."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(9, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 5)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def get_activations(self, x):
        """Return first hidden layer activations."""
        return F.relu(self.fc1(x))


class FullAdderBit(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_activations(self, x):
        return F.relu(self.fc1(x))


class Compositional4BitAdder(nn.Module):
    """Compositional architecture with activation extraction."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.bit0 = FullAdderBit(hidden_size)
        self.bit1 = FullAdderBit(hidden_size)
        self.bit2 = FullAdderBit(hidden_size)
        self.bit3 = FullAdderBit(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        a, b, cin = x[:, 0:4], x[:, 4:8], x[:, 8:9]

        out0 = self.bit0(torch.cat([a[:, 0:1], b[:, 0:1], cin], dim=1))
        s0, c0 = out0[:, 0:1], out0[:, 1:2]

        out1 = self.bit1(torch.cat([a[:, 1:2], b[:, 1:2], c0], dim=1))
        s1, c1 = out1[:, 0:1], out1[:, 1:2]

        out2 = self.bit2(torch.cat([a[:, 2:3], b[:, 2:3], c1], dim=1))
        s2, c2 = out2[:, 0:1], out2[:, 1:2]

        out3 = self.bit3(torch.cat([a[:, 3:4], b[:, 3:4], c2], dim=1))
        s3, c3 = out3[:, 0:1], out3[:, 1:2]

        return torch.cat([s0, s1, s2, s3, c3], dim=1)

    def get_activations(self, x):
        """Return concatenated activations from all bit adders."""
        a, b, cin = x[:, 0:4], x[:, 4:8], x[:, 8:9]

        # Bit 0
        act0 = self.bit0.get_activations(torch.cat([a[:, 0:1], b[:, 0:1], cin], dim=1))
        out0 = self.bit0(torch.cat([a[:, 0:1], b[:, 0:1], cin], dim=1))
        c0 = out0[:, 1:2]

        # Bit 1
        act1 = self.bit1.get_activations(torch.cat([a[:, 1:2], b[:, 1:2], c0], dim=1))
        out1 = self.bit1(torch.cat([a[:, 1:2], b[:, 1:2], c0], dim=1))
        c1 = out1[:, 1:2]

        # Bit 2
        act2 = self.bit2.get_activations(torch.cat([a[:, 2:3], b[:, 2:3], c1], dim=1))
        out2 = self.bit2(torch.cat([a[:, 2:3], b[:, 2:3], c1], dim=1))
        c2 = out2[:, 1:2]

        # Bit 3
        act3 = self.bit3.get_activations(torch.cat([a[:, 3:4], b[:, 3:4], c2], dim=1))

        return torch.cat([act0, act1, act2, act3], dim=1)  # (batch, 64)


class OnlineDeltaObserver(nn.Module):
    """
    Delta Observer that trains concurrently with task models.

    Same architecture as original, but trained online.
    """

    def __init__(self, mono_dim=64, comp_dim=64, latent_dim=16):
        super().__init__()

        self.mono_encoder = nn.Sequential(
            nn.Linear(mono_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.comp_encoder = nn.Sequential(
            nn.Linear(comp_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.shared_encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, latent_dim),
        )

        self.mono_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, mono_dim),
        )

        self.comp_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, comp_dim),
        )

        self.bit_classifier = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

        self.carry_regressor = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.latent_dim = latent_dim

    def encode(self, mono_act, comp_act):
        mono_enc = self.mono_encoder(mono_act)
        comp_enc = self.comp_encoder(comp_act)
        joint = torch.cat([mono_enc, comp_enc], dim=-1)
        return self.shared_encoder(joint)

    def forward(self, mono_act, comp_act):
        latent = self.encode(mono_act, comp_act)

        mono_recon = self.mono_decoder(latent)
        comp_recon = self.comp_decoder(latent)
        bit_logits = self.bit_classifier(latent)
        carry_pred = self.carry_regressor(latent)

        return {
            'latent': latent,
            'mono_recon': mono_recon,
            'comp_recon': comp_recon,
            'bit_logits': bit_logits,
            'carry_pred': carry_pred,
        }


# =============================================================================
# Training
# =============================================================================

def get_scheduler(optimizer, warmup_steps, total_steps, peak_lr, min_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / peak_lr, cosine)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_online(epochs=200, batch_size=32):
    """Train all three models concurrently."""

    print("=" * 80)
    print("PHASE 1: ONLINE DELTA OBSERVER TRAINING")
    print("=" * 80)

    # Generate data
    inputs, outputs, carry_counts, bit_positions = generate_dataset()
    n_samples = len(inputs)
    n_batches = n_samples // batch_size

    print(f"\nDataset: {n_samples} samples")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    # Create models
    mono_model = Monolithic4BitAdder(hidden_size=64)
    comp_model = Compositional4BitAdder(hidden_size=16)
    observer = OnlineDeltaObserver(mono_dim=64, comp_dim=64, latent_dim=16)

    print(f"\nModels created:")
    print(f"  Mono params: {sum(p.numel() for p in mono_model.parameters()):,}")
    print(f"  Comp params: {sum(p.numel() for p in comp_model.parameters()):,}")
    print(f"  Observer params: {sum(p.numel() for p in observer.parameters()):,}")

    # Optimizers
    mono_opt = optim.Adam(mono_model.parameters(), lr=ALPHA)
    comp_opt = optim.Adam(comp_model.parameters(), lr=ALPHA)
    obs_opt = optim.Adam(observer.parameters(), lr=0.001)

    total_steps = n_batches * epochs
    warmup_steps = n_batches * 10

    mono_sched = get_scheduler(mono_opt, warmup_steps, total_steps, ALPHA, ALPHA/1000)
    comp_sched = get_scheduler(comp_opt, warmup_steps, total_steps, ALPHA, ALPHA/1000)
    obs_sched = optim.lr_scheduler.CosineAnnealingLR(obs_opt, epochs)

    # Training log
    log = {
        'epochs': [],
        'start_time': datetime.now().isoformat(),
    }

    # Activation history (for trajectory analysis)
    activation_snapshots = []

    print("\n" + "-" * 80)
    print("Training...")
    print("-" * 80)

    for epoch in range(epochs):
        mono_model.train()
        comp_model.train()
        observer.train()

        indices = torch.randperm(n_samples)

        epoch_mono_loss = 0
        epoch_comp_loss = 0
        epoch_obs_loss = 0
        mono_correct = 0
        comp_correct = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = inputs[batch_idx]
            batch_y = outputs[batch_idx]
            batch_carry = carry_counts[batch_idx]
            batch_bit = bit_positions[batch_idx]

            # === Task Models Forward ===
            mono_out = mono_model(batch_x)
            comp_out = comp_model(batch_x)

            # Task losses
            mono_loss = F.mse_loss(mono_out, batch_y)
            comp_loss = F.mse_loss(comp_out, batch_y)

            # Task backward (before getting activations to avoid graph issues)
            mono_opt.zero_grad()
            mono_loss.backward()
            mono_opt.step()
            mono_sched.step()

            comp_opt.zero_grad()
            comp_loss.backward()
            comp_opt.step()
            comp_sched.step()

            # === Observer Forward (with detached activations) ===
            with torch.no_grad():
                mono_act = mono_model.get_activations(batch_x)
                comp_act = comp_model.get_activations(batch_x)

            obs_out = observer(mono_act, comp_act)

            # Observer losses
            recon_loss = (
                F.mse_loss(obs_out['mono_recon'], mono_act) +
                F.mse_loss(obs_out['comp_recon'], comp_act)
            )
            class_loss = F.cross_entropy(obs_out['bit_logits'], batch_bit)
            carry_loss = F.mse_loss(obs_out['carry_pred'].squeeze(), batch_carry.float())

            obs_loss = recon_loss + class_loss + 0.1 * carry_loss

            obs_opt.zero_grad()
            obs_loss.backward()
            obs_opt.step()

            # Accumulate
            epoch_mono_loss += mono_loss.item()
            epoch_comp_loss += comp_loss.item()
            epoch_obs_loss += obs_loss.item()

            # Accuracy
            mono_correct += (torch.round(mono_out).long() == batch_y.long()).all(dim=1).sum().item()
            comp_correct += (torch.round(comp_out).long() == batch_y.long()).all(dim=1).sum().item()

        obs_sched.step()

        # Normalize
        epoch_mono_loss /= n_batches
        epoch_comp_loss /= n_batches
        epoch_obs_loss /= n_batches
        mono_acc = mono_correct / n_samples
        comp_acc = comp_correct / n_samples

        # Snapshot activations periodically (for trajectory analysis)
        if epoch % 10 == 0 or epoch < 20:
            with torch.no_grad():
                mono_act_all = mono_model.get_activations(inputs)
                comp_act_all = comp_model.get_activations(inputs)
                latent_all = observer.encode(mono_act_all, comp_act_all)
                activation_snapshots.append({
                    'epoch': epoch,
                    'latent': latent_all.numpy().copy(),
                })

        log['epochs'].append({
            'epoch': epoch + 1,
            'mono_loss': epoch_mono_loss,
            'comp_loss': epoch_comp_loss,
            'obs_loss': epoch_obs_loss,
            'mono_acc': mono_acc,
            'comp_acc': comp_acc,
        })

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Mono: {mono_acc:.4f} | "
                  f"Comp: {comp_acc:.4f} | "
                  f"Obs Loss: {epoch_obs_loss:.4f}")

    print("-" * 80)
    print("Training complete.")

    # === Final Evaluation ===
    print("\n" + "=" * 80)
    print("EXTRACTING FINAL LATENT REPRESENTATIONS")
    print("=" * 80)

    observer.eval()
    mono_model.eval()
    comp_model.eval()

    with torch.no_grad():
        mono_act_final = mono_model.get_activations(inputs)
        comp_act_final = comp_model.get_activations(inputs)
        latent_final = observer.encode(mono_act_final, comp_act_final)

    # Save everything
    save_dir = 'data'
    np.savez(
        f'{save_dir}/online_observer_latents.npz',
        latents=latent_final.numpy(),
        carry_counts=carry_counts.numpy(),
        bit_positions=bit_positions.numpy(),
        mono_activations=mono_act_final.numpy(),
        comp_activations=comp_act_final.numpy(),
    )
    print(f"Saved: {save_dir}/online_observer_latents.npz")

    # Save trajectory
    np.savez(
        f'{save_dir}/online_observer_trajectory.npz',
        snapshots=[s['latent'] for s in activation_snapshots],
        epochs=[s['epoch'] for s in activation_snapshots],
    )
    print(f"Saved: {save_dir}/online_observer_trajectory.npz")

    # Save models
    torch.save(mono_model.state_dict(), 'models/online_mono_final.pt')
    torch.save(comp_model.state_dict(), 'models/online_comp_final.pt')
    torch.save(observer.state_dict(), 'models/online_observer_final.pt')
    print("Saved model weights.")

    log['end_time'] = datetime.now().isoformat()
    with open('journal/online_training_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    return latent_final.numpy(), carry_counts.numpy()


def evaluate_online_vs_pca(latents, carry_counts):
    """Compare online observer to PCA baseline."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    print("\n" + "=" * 80)
    print("ONLINE OBSERVER vs PCA BASELINE")
    print("=" * 80)

    # Online observer metrics
    reg = LinearRegression()
    reg.fit(latents, carry_counts)
    online_r2 = reg.score(latents, carry_counts)
    online_sil = silhouette_score(latents, carry_counts)

    # Load original data for PCA comparison
    orig_data = np.load('data/delta_observer_dataset.npz')
    mono_act = orig_data['mono_activations']
    comp_act = orig_data['comp_activations']
    combined = np.concatenate([mono_act, comp_act], axis=1)

    pca = PCA(n_components=16)
    pca_latents = pca.fit_transform(combined)

    reg_pca = LinearRegression()
    reg_pca.fit(pca_latents, carry_counts)
    pca_r2 = reg_pca.score(pca_latents, carry_counts)
    pca_sil = silhouette_score(pca_latents, carry_counts)

    # Post-hoc observer (from original data)
    orig_latents = np.load('data/delta_latent_umap.npz')['latents']
    reg_posthoc = LinearRegression()
    reg_posthoc.fit(orig_latents, carry_counts)
    posthoc_r2 = reg_posthoc.score(orig_latents, carry_counts)
    posthoc_sil = silhouette_score(orig_latents, carry_counts)

    print(f"\n{'Method':<25} {'R²':>12} {'Silhouette':>12}")
    print("-" * 50)
    print(f"{'Online Observer':<25} {online_r2:>12.4f} {online_sil:>12.4f}")
    print(f"{'Post-hoc Observer':<25} {posthoc_r2:>12.4f} {posthoc_sil:>12.4f}")
    print(f"{'PCA Baseline':<25} {pca_r2:>12.4f} {pca_sil:>12.4f}")

    # Verdict
    print("\n" + "-" * 50)
    if online_r2 > pca_r2 + 0.01:
        print(f"✓ Online Observer beats PCA by {online_r2 - pca_r2:.4f}")
        verdict = "PASS"
    elif online_r2 > pca_r2:
        print(f"~ Online Observer slightly better than PCA (+{online_r2 - pca_r2:.4f})")
        verdict = "MARGINAL"
    else:
        print(f"✗ Online Observer does not beat PCA ({online_r2:.4f} vs {pca_r2:.4f})")
        verdict = "FAIL"

    return {
        'online_r2': online_r2,
        'online_silhouette': online_sil,
        'posthoc_r2': posthoc_r2,
        'posthoc_silhouette': posthoc_sil,
        'pca_r2': pca_r2,
        'pca_silhouette': pca_sil,
        'verdict': verdict,
    }


if __name__ == "__main__":
    latents, carry_counts = train_online(epochs=200, batch_size=32)
    results = evaluate_online_vs_pca(latents, carry_counts)

    # Save results
    with open('journal/online_observer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to journal/online_observer_results.json")
